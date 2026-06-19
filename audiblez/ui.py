#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A simple wxWidgets UI for audiblez

import numpy as np
import soundfile
import threading
import platform
import subprocess
import io
import os
import wx
from wx.lib.newevent import NewEvent
from wx.lib.scrolledpanel import ScrolledPanel
from PIL import Image
from tempfile import NamedTemporaryFile
from pathlib import Path

from audiblez.voices import (voices, flags, VOICE_QUALITY, RECOMMENDED_VOICES,
                             PRESET_BLEND_INFO, DEFAULT_VOICE, parse_voice_spec, grade_rank,
                             voice_with_grade)
from audiblez import backends

EVENTS = {
    'CORE_STARTED': NewEvent(),
    'CORE_PROGRESS': NewEvent(),
    'CORE_CHAPTER_STARTED': NewEvent(),
    'CORE_CHAPTER_FINISHED': NewEvent(),
    'CORE_FINISHED': NewEvent()
}

border = 5


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        screen_width, screen_h = wx.GetDisplaySize()
        # Cap the default size so it doesn't open enormous on large/Retina displays.
        self.window_width = min(1400, int(screen_width * 0.7))
        window_height = min(900, int(screen_h * 0.8))
        super().__init__(parent, title=title, size=(self.window_width, window_height))
        self.SetMinSize((900, 600))
        self.chapters_panel = None
        self.preview_threads = []
        self.selected_chapter = None
        self.selected_book = None
        self.synthesis_in_progress = False

        self.Bind(EVENTS['CORE_STARTED'][1], self.on_core_started)
        self.Bind(EVENTS['CORE_CHAPTER_STARTED'][1], self.on_core_chapter_started)
        self.Bind(EVENTS['CORE_CHAPTER_FINISHED'][1], self.on_core_chapter_finished)
        self.Bind(EVENTS['CORE_PROGRESS'][1], self.on_core_progress)
        self.Bind(EVENTS['CORE_FINISHED'][1], self.on_core_finished)

        self.create_menu()
        self.create_layout()
        # Close any resident engine (the MOSS child holds VRAM) when the window closes, so it
        # isn't orphaned past the GUI session.
        self.Bind(wx.EVT_CLOSE, self.on_window_close)
        self.Centre()
        self.Show(True)
        if Path('../epub/lewis.epub').exists(): self.open_epub('../epub/lewis.epub')

    def create_menu(self):
        menubar = wx.MenuBar()
        file_menu = wx.Menu()
        open_item = wx.MenuItem(file_menu, wx.ID_OPEN, "&Open\tCtrl+O")
        file_menu.Append(open_item)
        self.Bind(wx.EVT_MENU, self.on_open, open_item)  # Bind the event

        exit_item = wx.MenuItem(file_menu, wx.ID_EXIT, "&Exit\tCtrl+Q")
        file_menu.Append(exit_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)

    def on_core_started(self, event):
        print('CORE_STARTED')
        self.progress_bar_label.Show()
        self.progress_bar.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Layout()
        self.eta_label.Show()
        if hasattr(self, 'done_label'): self.done_label.Hide()
        self.params_panel.Layout()
        self.synth_panel.Layout()

    def on_core_chapter_started(self, event):
        # print('CORE_CHAPTER_STARTED', event.chapter_index)
        self.set_table_chapter_status(event.chapter_index, "⏳ In Progress")

    def on_core_chapter_finished(self, event):
        # print('CORE_CHAPTER_FINISHED', event.chapter_index)
        self.set_table_chapter_status(event.chapter_index, "✅ Done")

    def on_core_progress(self, event):
        # print('CORE_PROGRESS', event.progress)
        self.progress_bar.SetValue(event.stats.progress)
        self.progress_bar_label.SetLabel(f"Synthesis Progress: {event.stats.progress}%")
        self.eta_label.SetLabel(f"Estimated Time Remaining: {event.stats.eta}")
        self.synth_panel.Layout()

    def on_core_finished(self, event):
        self.synthesis_in_progress = False
        out = self.output_folder_text_ctrl.GetValue()
        self.progress_bar.SetValue(100)
        self.progress_bar_label.SetLabel("Synthesis Progress: 100%")
        self.eta_label.Hide()
        self.done_label.SetLabel(f"✅ Done — audiobook saved to:\n{out}")
        self.done_label.Show()
        # Re-enable the controls disabled in on_start so another book can be generated.
        self.start_button.Enable()
        self.start_button.Show()
        self.params_panel.Enable()
        self.table.EnableCheckBoxes(True)
        self.synth_panel.Layout()
        self.open_folder_with_explorer(out)

    def create_layout(self):
        # Panels layout looks like this:
        # splitter
        #     splitter_left
        #         chapters_panel
        #     splitter_right
        #         center_panel
        #             text_area
        #         right_panel
        #             book_info_panel_box
        #                 book_info_panel
        #                     cover_bitmap
        #                     book_details_panel
        #             param_panel_box
        #                  param_panel
        #                      ...
        #             synth_panel_box
        #                  synth_panel
        #                      start_button
        #                      ...

        top_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_panel.SetSizer(top_sizer)

        # Open Epub button
        open_epub_button = wx.Button(top_panel, label="📁 Open EPUB")
        open_epub_button.Bind(wx.EVT_BUTTON, self.on_open)
        top_sizer.Add(open_epub_button, 0, wx.ALL, 5)

        # About button
        help_button = wx.Button(top_panel, label="ℹ️ About")
        help_button.Bind(wx.EVT_BUTTON, lambda event: self.about_dialog())
        top_sizer.Add(help_button, 0, wx.ALL, 5)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)

        # self.splitter = wx.SplitterWindow(self, -1)
        # self.splitter.SetSashGravity(0.9)
        self.splitter = wx.Panel(self)
        self.splitter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.splitter.SetSizer(self.splitter_sizer)

        # First-run hint shown in the empty area until an EPUB is opened.
        self.start_hint = wx.StaticText(
            self.splitter,
            label='👈  Click “📁 Open EPUB”  (or press ⌘O)  to load a book and begin.',
            style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.start_hint.SetFont(wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
        self.splitter_sizer.Add(self.start_hint, 1, wx.ALL | wx.EXPAND, 60)

        self.main_sizer.Add(top_panel, 0, wx.ALL | wx.EXPAND, 5)
        self.main_sizer.Add(self.splitter, 1, wx.EXPAND)

    def create_layout_for_ebook(self, splitter):
        splitter_left = wx.Panel(splitter, -1)
        splitter_right = wx.Panel(self.splitter)
        self.splitter_left, self.splitter_right = splitter_left, splitter_right
        self.splitter_sizer.Add(splitter_left, 2, wx.ALL | wx.EXPAND, 5)
        self.splitter_sizer.Add(splitter_right, 3, wx.ALL | wx.EXPAND, 5)

        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        splitter_left.SetSizer(self.left_sizer)

        # add center panel with large text area
        self.center_panel = wx.Panel(splitter_right)
        self.center_sizer = wx.BoxSizer(wx.VERTICAL)
        self.center_panel.SetSizer(self.center_sizer)
        self.text_area = wx.TextCtrl(self.center_panel, style=wx.TE_MULTILINE)
        font = wx.Font(14, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.text_area.SetFont(font)
        # On text change, update the extracted_text attribute of the selected_chapter:
        self.text_area.Bind(wx.EVT_TEXT, lambda event: setattr(self.selected_chapter, 'extracted_text', self.text_area.GetValue()))

        self.chapter_label = wx.StaticText(
            self.center_panel, label=f'Edit / Preview content for section "{self.selected_chapter.short_name}":')
        preview_button = wx.Button(self.center_panel, label="🔊 Preview")
        preview_button.Bind(wx.EVT_BUTTON, self.on_preview_chapter)

        self.center_sizer.Add(self.chapter_label, 0, wx.ALL, 5)
        self.center_sizer.Add(preview_button, 0, wx.ALL, 5)
        self.center_sizer.Add(self.text_area, 1, wx.ALL | wx.EXPAND, 5)

        splitter_right_sizer = wx.BoxSizer(wx.HORIZONTAL)
        splitter_right.SetSizer(splitter_right_sizer)

        self.create_right_panel(splitter_right)
        splitter_right_sizer.Add(self.center_panel, 1, wx.ALL | wx.EXPAND, 5)
        splitter_right_sizer.Add(self.right_panel, 1, wx.ALL | wx.EXPAND, 5)

    def about_dialog(self):
        msg = ("A simple tool to generate audiobooks from EPUB files using Kokoro-82M models\n" +
               "Distributed under the MIT License.\n\n" +
               "by Claudio Santini 2025\nand many contributors.\n\n" +
               "https://claudio.uk\n\n")
        wx.MessageBox(msg, "Audiblez")

    def create_right_panel(self, splitter_right):
        self.right_panel = wx.Panel(splitter_right)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(self.right_sizer)

        self.book_info_panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        book_info_panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.book_info_panel_box, "Book Details")
        self.book_info_panel_box.SetSizer(book_info_panel_box_sizer)
        self.right_sizer.Add(self.book_info_panel_box, 1, wx.ALL | wx.EXPAND, 5)

        self.book_info_panel = wx.Panel(self.book_info_panel_box, style=wx.BORDER_NONE)
        self.book_info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.book_info_panel.SetSizer(self.book_info_sizer)
        book_info_panel_box_sizer.Add(self.book_info_panel, 1, wx.ALL | wx.EXPAND, 5)

        # Add cover image
        self.cover_bitmap = wx.StaticBitmap(self.book_info_panel, -1)
        self.book_info_sizer.Add(self.cover_bitmap, 0, wx.ALL, 5)

        self.cover_bitmap.Refresh()
        self.book_info_panel.Refresh()
        self.book_info_panel.Layout()
        self.cover_bitmap.Layout()

        self.create_book_details_panel()
        self.create_params_panel()
        self.create_synthesis_panel()

    def create_book_details_panel(self):
        book_details_panel = wx.Panel(self.book_info_panel)
        book_details_sizer = wx.GridBagSizer(10, 10)
        book_details_panel.SetSizer(book_details_sizer)
        self.book_info_sizer.Add(book_details_panel, 1, wx.ALL | wx.EXPAND, 5)

        # Add title
        title_label = wx.StaticText(book_details_panel, label="Title:")
        title_text = wx.StaticText(book_details_panel, label=self.selected_book_title)
        book_details_sizer.Add(title_label, pos=(0, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(title_text, pos=(0, 1), flag=wx.ALL, border=5)

        # Add Author
        author_label = wx.StaticText(book_details_panel, label="Author:")
        author_text = wx.StaticText(book_details_panel, label=self.selected_book_author)
        book_details_sizer.Add(author_label, pos=(1, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(author_text, pos=(1, 1), flag=wx.ALL, border=5)

        # Add Total length
        length_label = wx.StaticText(book_details_panel, label="Total Length:")
        if not hasattr(self, 'document_chapters'):
            total_len = 0
        else:
            total_len = sum([len(c.extracted_text) for c in self.document_chapters])
        length_text = wx.StaticText(book_details_panel, label=f'{total_len:,} characters')
        book_details_sizer.Add(length_label, pos=(2, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(length_text, pos=(2, 1), flag=wx.ALL, border=5)

    def create_params_panel(self):
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Parameters")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.params_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.GridBagSizer(10, 10)
        panel.SetSizer(sizer)

        # Build one radio per backend actually available on this machine. The torch device
        # (or MLX engine) is selected later, once, in core.build_synthesizer on the worker
        # thread — not on every click — so selection here just records self.selected_backend.
        engine_label = wx.StaticText(panel, label="Engine:")
        engine_radio_panel = wx.Panel(panel)
        engine_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        engine_radio_panel.SetSizer(engine_radio_panel_sizer)
        # Render one radio per engine_choices() entry: the available backends are selectable;
        # MOSS, when not fully installed, is shown DISABLED with its reason ("MOSS (unavailable —
        # missing binary)") rather than silently hidden (PRD slice 1 / #7). The first choice is
        # always 'cpu' (always available), so the RB_GROUP anchor is never the disabled MOSS radio,
        # and the default selection comes from default_backend() (never the disabled radio).
        self.backend_radios = {}
        choices = backends.engine_choices()
        default = backends.default_backend()
        self.selected_backend = default
        for idx, choice in enumerate(choices):
            style = wx.RB_GROUP if idx == 0 else 0
            label = (choice.label if choice.available
                     else f'{choice.label} (unavailable — {choice.reason})')
            rb = wx.RadioButton(engine_radio_panel, label=label, style=style)
            if choice.available:
                rb.Bind(wx.EVT_RADIOBUTTON, lambda event, b=choice.id: self.on_select_backend(b))
            else:
                rb.Disable()  # present-but-disabled: visible, not selectable
            if choice.id == default:
                rb.SetValue(True)
            self.backend_radios[choice.id] = rb
            engine_radio_panel_sizer.Add(rb, 0, wx.ALL, 5)
        sizer.Add(engine_label, pos=(0, 0), flag=wx.ALL, border=border)
        sizer.Add(engine_radio_panel, pos=(0, 1), flag=wx.ALL, border=border)

        # Voice picker. Recommended single voices and curated "house narrator" blends are
        # surfaced first, then every voice annotated with its Kokoro quality grade and sorted
        # best-first within each language. A label->spec map keeps selection robust; the combo
        # is editable, so a user can also TYPE a custom blend (e.g. 'af_bella:60,af_heart:40').
        self.voice_label_to_spec = {}
        voice_choices = []

        def add_voice_choice(label, spec):
            if label in self.voice_label_to_spec:
                return
            self.voice_label_to_spec[label] = spec
            voice_choices.append(label)

        for v in RECOMMENDED_VOICES:
            add_voice_choice(f'⭐ {flags[v[0]]} {voice_with_grade(v)}', v)
        for name, desc in PRESET_BLEND_INFO.items():
            add_voice_choice(f'🎙️ {name} — {desc}', name)
        for code, vlist in voices.items():
            for v in sorted(vlist, key=lambda x: (grade_rank(VOICE_QUALITY.get(x, '')), x)):
                add_voice_choice(f'{flags[code]} {voice_with_grade(v)}', v)

        self.voice_label = voice_label = wx.StaticText(panel, label="Voice:")
        self.selected_voice = DEFAULT_VOICE
        default_label = next(lbl for lbl, spec in self.voice_label_to_spec.items() if spec == DEFAULT_VOICE)
        # Keep a handle: under the MOSS engine this dropdown is greyed out (MOSS ignores the
        # Kokoro voice — slice 2 / #8), re-enabled when switching back to a Kokoro backend.
        self.voice_dropdown = voice_dropdown = wx.ComboBox(panel, choices=voice_choices, value=default_label)
        voice_dropdown.Bind(wx.EVT_COMBOBOX, self.on_select_voice)
        voice_dropdown.Bind(wx.EVT_TEXT, self.on_select_voice)  # catch typed custom blends too
        audition_button = wx.Button(panel, label="🗣️ Audition voice")
        audition_button.Bind(wx.EVT_BUTTON, self.on_audition_voice)
        lexicon_button = wx.Button(panel, label="📖 Pronunciations")
        lexicon_button.Bind(wx.EVT_BUTTON, self.on_edit_lexicon)
        sizer.Add(voice_label, pos=(1, 0), flag=wx.ALL, border=border)
        sizer.Add(voice_dropdown, pos=(1, 1), flag=wx.ALL, border=border)
        sizer.Add(audition_button, pos=(1, 2), flag=wx.ALL, border=border)
        sizer.Add(lexicon_button, pos=(1, 3), flag=wx.ALL, border=border)

        # Speed: a spin control bounded to 0.5–2.0 (cannot accept invalid input)
        speed_label = wx.StaticText(panel, label="Speed:")
        self.selected_speed = 1.0
        speed_spin = wx.SpinCtrlDouble(panel, value="1.0", min=0.5, max=2.0, inc=0.1)
        speed_spin.SetDigits(1)
        speed_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_select_speed)
        sizer.Add(speed_label, pos=(2, 0), flag=wx.ALL, border=border)
        sizer.Add(speed_spin, pos=(2, 1), flag=wx.ALL, border=border)

        # Add file dialog selector to select output folder
        output_folder_label = wx.StaticText(panel, label="Output Folder:")
        self.output_folder_text_ctrl = wx.TextCtrl(panel, value=os.path.abspath('.'))
        self.output_folder_text_ctrl.SetEditable(False)
        # self.output_folder_text_ctrl.SetMinSize((200, -1))
        output_folder_button = wx.Button(panel, label="📂 Select")
        output_folder_button.Bind(wx.EVT_BUTTON, self.open_output_folder_dialog)
        sizer.Add(output_folder_label, pos=(3, 0), flag=wx.ALL, border=border)
        sizer.Add(self.output_folder_text_ctrl, pos=(3, 1), flag=wx.ALL | wx.EXPAND, border=border)
        sizer.Add(output_folder_button, pos=(4, 1), flag=wx.ALL, border=border)

        # MOSS-only Clone voice control (slice 3 / #9): pick a reference WAV to narrate the whole
        # book in that voice, or leave empty for MOSS's built-in voice. Built always (so engine
        # switching only Show/Hide-s it — no layout rebuild) and revealed by _sync_engine_controls
        # only when the llamacpp engine is selected. Disabled with a reason when the MOSS encoder
        # GGUF is absent (cloning requires it).
        self.clone_ref = None  # absolute path to the reference WAV, or None for built-in voice
        self.clone_label = wx.StaticText(panel, label="Clone voice:")
        self.clone_value = wx.StaticText(panel, label="Built-in voice")
        self.clone_choose_button = wx.Button(panel, label="🎙️ Choose reference…")
        self.clone_choose_button.Bind(wx.EVT_BUTTON, self.on_choose_clone_ref)
        self.clone_clear_button = wx.Button(panel, label="✖ Clear")
        self.clone_clear_button.Bind(wx.EVT_BUTTON, self.on_clear_clone_ref)
        sizer.Add(self.clone_label, pos=(5, 0), flag=wx.ALL, border=border)
        sizer.Add(self.clone_value, pos=(5, 1), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=border)
        sizer.Add(self.clone_choose_button, pos=(5, 2), flag=wx.ALL, border=border)
        sizer.Add(self.clone_clear_button, pos=(5, 3), flag=wx.ALL, border=border)
        self._clone_widgets = (self.clone_label, self.clone_value,
                               self.clone_choose_button, self.clone_clear_button)

        # Set the initial enabled/shown state to match the default engine (Kokoro unless MOSS is
        # fully installed): greys the voice dropdown + reveals the clone control only under MOSS.
        self._sync_engine_controls()

    def create_synthesis_panel(self):
        # Think and identify layout issue with the folling code
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Generation Status")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.synth_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        # Trailer: audition the whole book (chapter detection + voice) before committing.
        self.trailer_button = wx.Button(panel, label="🎬 Preview whole book (trailer)")
        self.trailer_button.Bind(wx.EVT_BUTTON, self.on_preview_book_trailer)
        sizer.Add(self.trailer_button, 0, wx.ALL, 5)

        # Add Start button
        self.start_button = wx.Button(panel, label="🚀 Start Audiobook Synthesis")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        sizer.Add(self.start_button, 0, wx.ALL, 5)

        # Add Progress Bar label:
        self.progress_bar_label = wx.StaticText(panel, label="Synthesis Progress:")
        sizer.Add(self.progress_bar_label, 0, wx.ALL, 5)
        self.progress_bar = wx.Gauge(panel, range=100, style=wx.GA_PROGRESS)
        self.progress_bar.SetMinSize((-1, 30))
        sizer.Add(self.progress_bar, 0, wx.ALL | wx.EXPAND, 5)
        self.progress_bar_label.Hide()
        self.progress_bar.Hide()

        # Add ETA Label
        self.eta_label = wx.StaticText(panel, label="Estimated Time Remaining: ")
        self.eta_label.Hide()
        sizer.Add(self.eta_label, 0, wx.ALL, 5)

        # Completion banner (shown when synthesis finishes)
        self.done_label = wx.StaticText(panel, label="")
        self.done_label.SetForegroundColour(wx.Colour(40, 160, 70))
        self.done_label.Hide()
        sizer.Add(self.done_label, 0, wx.ALL, 5)

    def open_output_folder_dialog(self, event):
        with wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            output_folder = dialog.GetPath()
            print(f"Selected output folder: {output_folder}")
            self.output_folder_text_ctrl.SetValue(output_folder)

    def on_select_voice(self, event):
        # Map the chosen label back to its voice spec. If the user typed something not in the
        # map, accept it as a raw spec only when it parses (e.g. 'af_bella:60,af_heart:40');
        # otherwise keep the previous selection (ignores half-typed input).
        choice = event.GetString()
        spec = self.voice_label_to_spec.get(choice)
        if spec is None:
            try:
                parse_voice_spec(choice)
                spec = choice
            except ValueError:
                return
        self.selected_voice = spec

    def on_select_speed(self, event):
        self.selected_speed = event.GetValue()  # SpinCtrlDouble -> bounded float
        print('Selected speed', self.selected_speed)

    def on_select_backend(self, backend_id):
        """Record the chosen engine and resync the engine-dependent controls.

        Bound to each *available* engine radio. Speed is intentionally NOT touched — it stays
        live under every engine (MOSS applies it downstream via atempo)."""
        self.selected_backend = backend_id
        self._sync_engine_controls()

    def _is_moss_engine(self):
        return backends.BACKENDS[getattr(self, 'selected_backend', 'cpu')].engine == 'llamacpp'

    def _sync_engine_controls(self):
        """Make the voice + clone controls match the selected engine (slices 2 & 3 / #8, #9).

        Under MOSS (engine 'llamacpp'): grey the Kokoro voice dropdown — MOSS ignores it — and
        reveal the Clone voice control (disabled with a reason when the encoder GGUF is absent).
        Under any Kokoro backend: re-enable the dropdown and hide the clone control. The dropdown
        never vanishes (no layout reshuffle); only the clone row is shown/hidden."""
        moss = self._is_moss_engine()
        # Voice dropdown: greyed under MOSS with an explanatory label (slice 2).
        self.voice_dropdown.Enable(not moss)
        self.voice_label.SetLabel("Voice (MOSS built-in):" if moss else "Voice:")
        self.voice_dropdown.SetToolTip(
            "MOSS uses its built-in voice — set a Clone reference to change it." if moss else "")
        # Clone control: shown only under MOSS; gated on the encoder GGUF (slice 3).
        encoder_present = backends.moss_paths().get('encoder') is not None
        for w in self._clone_widgets:
            w.Show(moss)
        if moss:
            self.clone_choose_button.Enable(encoder_present)
            self.clone_clear_button.Enable(encoder_present and self.clone_ref is not None)
            if not encoder_present:
                self.clone_value.SetLabel("Cloning unavailable — missing MOSS encoder model")
            else:
                self._refresh_clone_value()
        self.params_panel.Layout()

    def _refresh_clone_value(self):
        """Update the clone-selection display + Clear-button state from self.clone_ref."""
        if self.clone_ref:
            self.clone_value.SetLabel(os.path.basename(self.clone_ref))
        else:
            self.clone_value.SetLabel("Built-in voice")
        self.clone_clear_button.Enable(self.clone_ref is not None)

    def on_choose_clone_ref(self, event):
        """Pick a reference WAV via a file dialog, with up-front validation (slice 3 / #9).

        Light validation only (exists + readable WAV header) so a bad pick raises a clean dialog
        now instead of a deep traceback hours into a render; deeper checks defer to synth time."""
        with wx.FileDialog(self, "Choose a reference WAV to clone", wildcard="WAV files (*.wav)|*.wav",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = dialog.GetPath()
        try:
            self._validate_clone_ref(path)
        except Exception as e:
            wx.MessageBox(f"Not a usable reference WAV:\n{e}", "Invalid clone reference",
                          wx.OK | wx.ICON_ERROR)
            return
        self.clone_ref = os.path.abspath(path)
        self._refresh_clone_value()

    @staticmethod
    def _validate_clone_ref(path):
        """Raise (with a human message) unless ``path`` is an existing, readable WAV header."""
        if not os.path.isfile(path):
            raise FileNotFoundError("file does not exist")
        # soundfile reads only the header in info() — cheap, and it rejects non-WAV/corrupt files.
        info = soundfile.info(path)
        if info.frames <= 0:
            raise ValueError("file contains no audio frames")

    def on_clear_clone_ref(self, event):
        """Revert to MOSS's built-in voice."""
        self.clone_ref = None
        self._refresh_clone_value()

    def get_selected_clone_ref(self):
        """The active clone reference: only meaningful under MOSS (Kokoro ignores it)."""
        return self.clone_ref if self._is_moss_engine() else None

    def open_epub(self, file_path):
        # Cleanup previous layout (and the first-run hint) before (re)building.
        self.splitter.DestroyChildren()
        self.splitter_sizer.Clear()
        self.chapters_panel = None

        self.selected_file_path = file_path
        print(f"Opening file: {file_path}")  # Do something with the filepath (e.g., parse the EPUB)

        from ebooklib import epub
        from audiblez.core import (find_document_chapters_and_extract_texts, find_good_chapters,
                                   find_cover, extract_book_metadata)
        book = epub.read_epub(file_path)
        self.selected_book_title, self.selected_book_author = extract_book_metadata(book)
        self.selected_book = book

        self.document_chapters = find_document_chapters_and_extract_texts(book)
        good_chapters = find_good_chapters(self.document_chapters)
        good_ids = {id(c) for c in good_chapters}
        self.selected_chapter = good_chapters[0]
        for chapter in self.document_chapters:
            chapter.short_name = chapter.get_name().replace('.xhtml', '').replace('xhtml/', '').replace('.html', '').replace('Text/', '')
            chapter.is_selected = id(chapter) in good_ids

        self.create_layout_for_ebook(self.splitter)
        # Pre-fill the editor with the first selected chapter's text.
        self.text_area.SetValue(self.selected_chapter.extracted_text)

        # Update Cover
        cover = find_cover(book)
        if cover is not None:
            pil_image = Image.open(io.BytesIO(cover.content))
            wx_img = wx.Image(pil_image.size[0], pil_image.size[1])
            wx_img.SetData(pil_image.convert("RGB").tobytes())
            cover_h = 200
            cover_w = int(cover_h * pil_image.size[0] / pil_image.size[1])
            wx_img.Rescale(cover_w, cover_h)
            self.cover_bitmap.SetBitmap(wx_img.ConvertToBitmap())
            self.cover_bitmap.SetMaxSize((200, cover_h))

        chapters_panel = self.create_chapters_table_panel(good_chapters)

        #  chapters_panel to left_sizer, or replace if it exists already
        if self.chapters_panel:
            self.left_sizer.Replace(self.chapters_panel, chapters_panel)
            self.chapters_panel.Destroy()
            self.chapters_panel = chapters_panel
        else:
            self.left_sizer.Add(chapters_panel, 1, wx.ALL | wx.EXPAND, 5)
            self.chapters_panel = chapters_panel

        # These two are very important:
        self.splitter_left.Layout()
        self.splitter_right.Layout()
        self.splitter.Layout()

    def on_table_checked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = True

    def on_table_unchecked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = False

    def on_table_selected(self, event):
        chapter = self.document_chapters[event.GetIndex()]
        print('Selected', event.GetIndex(), chapter.short_name)
        self.selected_chapter = chapter
        self.text_area.SetValue(chapter.extracted_text)
        self.chapter_label.SetLabel(f'Edit / Preview content for section "{chapter.short_name}":')

    def create_chapters_table_panel(self, good_chapters):
        panel = ScrolledPanel(self.splitter_left, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        self.table = table = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        table.InsertColumn(0, "Included")
        table.InsertColumn(1, "Chapter Name")
        table.InsertColumn(2, "Chapter Length")
        table.InsertColumn(3, "Status")
        table.SetColumnWidth(0, 80)
        table.SetColumnWidth(1, 150)
        table.SetColumnWidth(2, 150)
        table.SetColumnWidth(3, 100)
        table.SetSize((250, -1))
        table.EnableCheckBoxes()
        table.Bind(wx.EVT_LIST_ITEM_CHECKED, self.on_table_checked)
        table.Bind(wx.EVT_LIST_ITEM_UNCHECKED, self.on_table_unchecked)
        table.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_table_selected)

        good_ids = {id(c) for c in good_chapters}
        for i, chapter in enumerate(self.document_chapters):
            auto_selected = id(chapter) in good_ids
            table.Append(['', chapter.short_name, f"{len(chapter.extracted_text):,}"])
            if auto_selected: table.CheckItem(i)

        title_text = wx.StaticText(panel, label="Select chapters to include in the audiobook:")
        sizer.Add(title_text, 0, wx.ALL, 5)
        sizer.Add(table, 1, wx.ALL | wx.EXPAND, 5)
        return panel

    def get_selected_voice(self):
        return self.selected_voice  # already a resolved spec (id, preset name, or blend)

    def get_selected_speed(self):
        return float(self.selected_speed)

    def _spawn_player(self, button, idle_label, produce_path, busy_label="⏳"):
        """Run produce_path(core) -> wav path on a daemon thread, ffplay it, then clean up.

        Shared by chapter preview, voice audition, and the book trailer so the off-UI
        thread machinery (never join() on the UI thread — it freezes the GUI) lives once.
        produce_path returns a path to a temporary wav to play, or None to skip.

        ``busy_label`` shows while working; the MOSS callers pass a "loading model" variant for
        the first (cold) call so the seconds-long resident-child warm-up doesn't look like a hang."""
        button.SetLabel(busy_label)
        button.Disable()

        def restore():
            button.SetLabel(idle_label)
            button.Enable()

        def work():
            path = None
            try:
                import audiblez.core as core
                path = produce_path(core)
                if path:
                    subprocess.run(['ffplay', '-autoexit', '-nodisp', str(path)])
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                wx.CallAfter(restore)  # GUI mutations must run on the main thread

        self.preview_threads = [t for t in self.preview_threads if t.is_alive()]
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.preview_threads.append(thread)

    def _cached_synth(self, core, voice, backend, clone_ref=None):
        """Reuse the built synthesizer across preview/audition clicks; building one reloads
        the whole Kokoro model (seconds + GPU memory) — for MOSS it spawns a resident child,
        so the cold start (~2.7s) is paid once per GUI session. Rebuild only when voice/backend
        change, and tear down the PREVIOUS engine first so a MOSS child isn't orphaned (it
        holds ~8 GB of VRAM) when the user switches voice/backend.

        ``clone_ref`` is part of the cache key (slice 3 / #9): changing the clone reference must
        tear down the stale MOSS child and respawn it encoding the NEW reference, else audition
        would replay the previous voice — a silent-wrongness the audition must not commit."""
        key = (voice, backend, clone_ref)
        if getattr(self, '_synth_cache_key', None) != key:
            self._close_cached_synth()
            self._synth = core.build_synthesizer(voice, backend, clone_ref=clone_ref)
            self._synth_cache_key = key
        return self._synth

    def _close_cached_synth(self):
        """Tear down the cached engine (closes the resident MOSS child; no-op for Kokoro)."""
        synth = getattr(self, '_synth', None)
        if synth is not None:
            getattr(synth, 'close', lambda: None)()
        self._synth = None
        self._synth_cache_key = None

    def _synth_to_temp(self, core, text, voice, speed):
        """Synthesize text to a temp wav and return its path (or None if no audio)."""
        backend = getattr(self, 'selected_backend', 'cpu')
        clone_ref = self.get_selected_clone_ref()  # None unless MOSS + a reference chosen
        synth = self._cached_synth(core, voice, backend, clone_ref=clone_ref)
        core.load_spacy()
        audio_segments = core.gen_audio_segments(synth, text, voice=voice, speed=speed)
        if not audio_segments:
            return None
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            soundfile.write(tmp, np.concatenate(audio_segments), core.sample_rate)
            path = tmp.name
        # MOSS (engine='llamacpp') synthesizes at 1.0 — its synth IGNORES speed — so the audition
        # must be re-stretched via atempo to match what the real render/trailer produce (US-26);
        # without this, preview always plays at 1.0 (a silent speed mismatch). Kokoro baked speed
        # into the samples already, so this is MOSS-only. Best-effort: if ffmpeg is absent the
        # preview falls back to 1.0 with a warning rather than crashing the GUI.
        if core.backends.BACKENDS[backend].engine == 'llamacpp':
            try:
                core._apply_atempo(path, speed)
            except RuntimeError as e:
                print(f'Preview tempo not applied: {e}')
        return path

    def on_preview_chapter(self, event):
        button = event.GetEventObject()
        text = self.selected_chapter.extracted_text[:300]
        if len(text.strip()) == 0:
            return
        voice, speed = self.get_selected_voice(), self.get_selected_speed()
        self._spawn_player(button, "🔊 Preview",
                           lambda core: self._synth_to_temp(core, text, voice, speed))

    def on_audition_voice(self, event):
        # Speak the book's opening in the chosen voice — an audible casting step for the
        # otherwise-cryptic voice dropdown.
        button = event.GetEventObject()
        chapter = self.selected_chapter
        text = chapter.extracted_text[:300] if chapter else ''
        if len(text.strip()) == 0:
            return
        voice, speed = self.get_selected_voice(), self.get_selected_speed()
        self._spawn_player(button, "🗣️ Audition voice",
                           lambda core: self._synth_to_temp(core, text, voice, speed),
                           busy_label=self._audition_busy_label(voice))

    def _audition_busy_label(self, voice):
        """"⏳ loading model…" when the upcoming MOSS audition will pay the cold-start child
        spawn (no warm synth for this exact voice/backend/clone_ref), else a plain "⏳"."""
        if not self._is_moss_engine():
            return "⏳"
        key = (voice, self.selected_backend, self.get_selected_clone_ref())
        warm = getattr(self, '_synth_cache_key', None) == key
        return "⏳" if warm else "⏳ loading model…"

    def on_preview_book_trailer(self, event):
        # Render a short sampler of every selected chapter's opening before committing.
        button = event.GetEventObject()
        file_path = self.selected_file_path
        voice, speed = self.get_selected_voice(), self.get_selected_speed()
        backend = getattr(self, 'selected_backend', 'cpu')
        clone_ref = self.get_selected_clone_ref()  # MOSS clone (None for Kokoro/built-in)
        selected = [c for c in self.document_chapters if getattr(c, 'is_selected', False)] or None
        # Read on the UI thread; pass the SAME folder the lexicon editor saves to, so the
        # trailer auditions this book's edited pronunciations (not an empty cwd lexicon).
        output_folder = self.output_folder_text_ctrl.GetValue()

        def produce(core):
            import tempfile
            fd, name = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            result = core.make_trailer(file_path, voice, name, speed=speed, backend=backend,
                                       selected_chapters=selected, output_folder=output_folder,
                                       clone_ref=clone_ref)
            if not result:
                try:
                    os.unlink(name)
                except OSError:
                    pass
                return None
            return result

        self._spawn_player(button, "🎬 Preview whole book (trailer)", produce)

    def on_edit_lexicon(self, event):
        """Open a small JSON editor for this book's pronunciation lexicon (term -> respelling)."""
        import json
        from audiblez import lexicon
        path = lexicon.lexicon_path(self.selected_file_path, self.output_folder_text_ctrl.GetValue())
        current = lexicon.load_lexicon(path)
        if not current:
            # Seed candidates from the book so the editor isn't empty on first open. This
            # regexes the whole book, so show a busy cursor while it scans a large novel.
            with wx.BusyCursor():
                book_text = '\n'.join(c.extracted_text for c in getattr(self, 'document_chapters', []))
                current = lexicon.build_seed_lexicon(book_text)

        dialog = wx.Dialog(self, title='Pronunciation Lexicon', size=(520, 540))
        sizer = wx.BoxSizer(wx.VERTICAL)
        info = wx.StaticText(dialog, label='Set the respelling (value) for each term, then Save.\n'
                                           'Entries where the value equals the term are ignored.')
        sizer.Add(info, 0, wx.ALL, 8)
        text_ctrl = wx.TextCtrl(dialog, style=wx.TE_MULTILINE,
                                value=json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True))
        text_ctrl.SetFont(wx.Font(11, wx.MODERN, wx.NORMAL, wx.NORMAL))
        sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 8)
        sizer.Add(dialog.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        dialog.SetSizer(sizer)

        if dialog.ShowModal() == wx.ID_OK:
            try:
                mapping = json.loads(text_ctrl.GetValue())
                if not isinstance(mapping, dict):
                    raise ValueError('Lexicon must be a JSON object mapping "term" to "respelling".')
                lexicon.save_lexicon(path, mapping)
                wx.MessageBox(f'Saved {len(mapping)} term(s) to\n{path}', 'Pronunciation Lexicon')
            except (ValueError, json.JSONDecodeError) as e:
                wx.MessageBox(f'Could not save lexicon:\n{e}', 'Invalid JSON')
        dialog.Destroy()

    def on_start(self, event):
        self.synthesis_in_progress = True
        file_path = self.selected_file_path
        voice = self.get_selected_voice()
        speed = float(self.selected_speed)
        selected_chapters = [chapter for chapter in self.document_chapters if chapter.is_selected]
        self.start_button.Disable()
        self.params_panel.Disable()

        self.table.EnableCheckBoxes(False)
        for chapter_index, chapter in enumerate(self.document_chapters):
            if chapter in selected_chapters:
                self.set_table_chapter_status(chapter_index, "Planned")
                self.table.SetItem(chapter_index, 0, '✔️')

        print('Starting Audiobook Synthesis', dict(file_path=file_path, voice=voice, pick_manually=False, speed=speed))
        self.core_thread = CoreThread(params=dict(
            file_path=file_path, voice=voice, pick_manually=False, speed=speed,
            output_folder=self.output_folder_text_ctrl.GetValue(),
            selected_chapters=selected_chapters, backend=self.selected_backend,
            clone_ref=self.get_selected_clone_ref()))  # MOSS voice clone (None for Kokoro/built-in)
        self.core_thread.start()

    def on_open(self, event):
        with wx.FileDialog(self, "Open EPUB File", wildcard="*.epub", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            file_path = dialog.GetPath()
            print(f"Selected file: {file_path}")
            if not file_path:
                print('No filepath?')
                return
            if self.synthesis_in_progress:
                wx.MessageBox("Audiobook synthesis is still in progress. Please wait for it to finish.", "Audiobook Synthesis in Progress")
            else:
                wx.CallAfter(self.open_epub, file_path)

    def on_exit(self, event):
        self.Close()

    def on_window_close(self, event):
        # Release the cached engine (kills the resident MOSS child + frees its VRAM) before
        # the window is destroyed, then let the default close handling proceed.
        try:
            self._close_cached_synth()
        except Exception:
            pass
        event.Skip()

    def set_table_chapter_status(self, chapter_index, status):
        self.table.SetItem(chapter_index, 3, status)

    def open_folder_with_explorer(self, folder_path):
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', folder_path])
            elif platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', folder_path])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', folder_path])
        except Exception as e:
            print(e)


class CoreThread(threading.Thread):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        import audiblez.core as _core
        from audiblez.core import main
        # Resolve MossRunAborted defensively, mirroring cli.py: a stubbed/partial core that the
        # CLI tolerates would otherwise ImportError this GUI thread (the "GUI is the one that
        # gets missed" trap). The sentinel is never raised, so the handler simply never matches.
        class _UnreachableAbort(BaseException): pass
        MossRunAborted = getattr(_core, 'MossRunAborted', _UnreachableAbort)
        try:
            main(**self.params, post_event=self.post_event)
        # MossRunAborted is a BaseException (so it propagates through core's broad
        # `except Exception` silence-handlers up to here); catch it explicitly or the GUI
        # would lock — no error dialog, no re-enabled controls — when MOSS aborts loud.
        except (Exception, MossRunAborted) as e:
            import traceback
            traceback.print_exc()
            # Capture the message NOW: Python deletes the `except ... as e` binding when
            # the block exits, but show_error runs later via CallAfter (it would NameError
            # on `e`). error_msg is a normal local, so the closure can use it safely.
            error_msg = str(e)

            def show_error():
                win = wx.GetApp().GetTopWindow()
                win.synthesis_in_progress = False
                win.start_button.Enable()
                win.params_panel.Enable()
                wx.MessageBox(f"Audiobook generation failed:\n{error_msg}", "Audiblez Error")

            wx.CallAfter(show_error)

    def post_event(self, event_name, **kwargs):
        # eg. 'EVENT_CORE_PROGRESS' -> EventCoreProgress, EVENT_CORE_PROGRESS
        EventObject, EVENT_CODE = EVENTS[event_name]
        event_object = EventObject()
        for k, v in kwargs.items():
            setattr(event_object, k, v)
        wx.PostEvent(wx.GetApp().GetTopWindow(), event_object)


def main():
    print('Starting GUI...')
    app = wx.App(False)
    frame = MainWindow(None, "Audiblez - Generate Audiobooks from E-books")
    frame.Show(True)
    frame.Layout()
    app.SetTopWindow(frame)
    print('Done.')
    app.MainLoop()


if __name__ == '__main__':
    main()
