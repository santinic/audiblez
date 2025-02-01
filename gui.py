import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import threading
import io
from kokoro import KPipeline
import ebooklib
from PIL import Image, ImageTk


class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, str)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass


LANGUAGE_TO_FLAG = {
    "en-us": "🇺🇸",
    "en-gb": "🇬🇧",
    "fr-fr": "🇫🇷",
    "ja": "🇯🇵",
    "ko": "🇰🇷",
    "cmn": "🇨🇳"
}


def get_language_from_voice(voice):
    if voice.startswith("a"):
        return "en-us"
    elif voice.startswith("b"):
        return "en-gb"
    else:
        print("Voice not recognized.")
        exit(1)


def emojify_voice(voice):
    language = get_language_from_voice(voice)
    if language in LANGUAGE_TO_FLAG:
        return LANGUAGE_TO_FLAG[language] + " " + voice
    return voice


def deemojify_voice(voice):
    if voice[:2] in LANGUAGE_TO_FLAG.values():
        return voice[3:]
    return voice


def start_gui():
    root = tk.Tk()
    root.title('Audiblez')
    root.geometry('1200x900')
    root.resizable(False, False)

    voice_frame = tk.Frame(root)
    voice_frame.pack(pady=5, padx=5)

    # add a scale to set speed
    speed_label = tk.Label(voice_frame, text="Set speed:", font=('Arial', 12))
    speed_label.pack(side=tk.LEFT, pady=5, padx=5)

    def check_speed_range(event=None):
        try:
            value = float(speed_entry.get())
            if 0.5 <= value <= 2.0:
                speed_entry.configure(fg='white')
                return True
            else:
                speed_entry.configure(fg='red')
        except ValueError:
            speed_entry.configure(fg='red')
        return False

    speed_entry = tk.Entry(
        voice_frame,
        width=5,
        font=('Arial', 12)
    )
    speed_entry.insert(0, "1.0")
    speed_entry.pack(side=tk.LEFT, pady=5, padx=5)
    speed_entry.bind('<KeyRelease>', check_speed_range)
    
    # add a combo box with voice options
    voice_label = tk.Label(voice_frame, text="Select Voice:", font=('Arial', 12))
    voice_label.pack(side=tk.LEFT, pady=5, padx=5)

    # add a combo box with voice options
    from audiblez import voices
    voices = [emojify_voice(x) for x in voices]
    voice_combo = ttk.Combobox(
        voice_frame,
        values=voices,
        state="readonly",
        font=('Arial', 12)
    )
    voice_combo.set(voices[0])  # Set default selection
    voice_combo.pack(side=tk.LEFT, pady=10, padx=5)

    # ui element variables
    pil_image = Image.new('RGB', (200, 300), 'gray')
    cover_image = ImageTk.PhotoImage(pil_image)  # or use a default image
    cover_label = tk.Label(root, image=cover_image)
    chapters_by_name = []

    def resized_image(item):
        image_data = item.get_content()
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail((200, 300))
        ratio = min(200/image.width, 300/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        # Resize with high-quality resampling
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        # Create new image with gray background
        background = Image.new('RGB', (200, 300), 'gray')
        # Paste resized image centered
        offset = ((200 - new_size[0])//2, (300 - new_size[1])//2)
        background.paste(resized, offset)
        return ImageTk.PhotoImage(background)

    def get_cover_image(book):
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_COVER:
                return resized_image(item)
            if item.get_type() == ebooklib.ITEM_IMAGE:
                if 'cover' in item.get_name().lower():
                    return resized_image(item)
        return None

    def select_file():
        file_path = filedialog.askopenfilename(
            title='Select an epub file',
            filetypes=[('epub files', '*.epub')]
        )
        if file_path:
            file_label.config(text=file_path)
            book = ebooklib.epub.read_epub(file_path)
            cover_image_from_book = get_cover_image(book)
            if cover_image_from_book:
                cover_label.image = cover_image_from_book
                cover_label.configure(image=cover_image_from_book)
            else:
                cover_label.image = cover_image
                cover_label.configure(image=cover_image)
            
            # set chapters
            chapters_by_name.clear()
            chapters_listbox.delete(0, tk.END)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapters_by_name.append(item.get_name())
            chapters_by_name.sort()
            for chapter in chapters_by_name:
                chapters_listbox.insert(tk.END, chapter)
            
    
    def convert():
        def enable_controls():
            speed_entry.configure(state='normal')
            voice_combo.configure(state='normal')
        
        def run_conversion():
            try:
                from audiblez import main
                pipeline = KPipeline(lang_code=voice[0])
                chapters = [chapters_selected_listbox.get(i) for i in range(chapters_selected_listbox.size())]
                main(pipeline, file_path, voice, False, float(speed), chapters)
            finally:
                # Ensure controls are re-enabled even if an error occurs
                root.after(0, enable_controls)
        
        if not check_speed_range():
            warning = "Please enter a speed value between 0.5 and 2.0."
            print(warning)
            # create a warning message box to say this
            messagebox.showwarning("Warning", warning)

        if file_label.cget("text"):
            output_text.configure(state='normal')
            output_text.delete(1.0, tk.END)
            output_text.configure(state='disabled')
            # Redirect stdout to Text widget
            sys.stdout = TextRedirector(output_text)
            file_path = file_label.cget("text")
            voice = deemojify_voice(voice_combo.get())
            speed = speed_entry.get()
            speed_entry.configure(state='disabled')
            voice_combo.configure(state='disabled')
            threading.Thread(target=run_conversion).start()
            # when this thread finishes, re-enable the buttons

        else:
            warning = "Please select an epub file first."
            print(warning)
            # create a warning message box to say this
            messagebox.showwarning("Warning", warning)

    file_frame = tk.Frame(root)
    file_frame.pack(pady=5)

    file_button = tk.Button(
        file_frame,
        text='Select epub file',
        command=select_file,
        bg='white',
        fg='black',
        font=('Arial', 12)
    )
    file_button.pack(side=tk.LEFT, pady=5, padx=5)

    file_label = tk.Label(file_frame, text="")
    file_label.pack(side=tk.LEFT, pady=5)

    cover_label.image = cover_image  # Keep a reference to prevent garbage collection
    cover_label.pack(pady=10)

    # Create frame for listbox and scrollbar
    chapters_frame = ttk.Frame(root)
    chapters_frame.pack(expand=True, padx=10, pady=10)

    # Create left frame for chapters_listbox
    left_frame = ttk.Frame(chapters_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create middle frame for buttons
    middle_frame = ttk.Frame(chapters_frame)
    middle_frame.pack(side=tk.LEFT, padx=10)

    # Create right frame for selected chapters
    right_frame = ttk.Frame(chapters_frame)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Left listbox and scrollbar
    scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chapters_listbox = tk.Listbox(
        left_frame,
        selectmode=tk.MULTIPLE,
        yscrollcommand=scrollbar.set,
        font=('Arial', 12),
        width=60,
        height=10
    )
    chapters_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=chapters_listbox.yview)

    def on_add_select():
        selected = chapters_listbox.curselection()
        for i in selected:
            chapters_selected_listbox.insert(tk.END, chapters_listbox.get(i))

    def on_remove_select():
        selected = chapters_selected_listbox.curselection()
        for i in selected:
            chapters_selected_listbox.delete(i)
    
    def on_add_all():
        for i in range(chapters_listbox.size()):
            chapters_selected_listbox.insert(tk.END, chapters_listbox.get(i))


    # Middle buttons
    add_button = tk.Button(
        middle_frame,
        text='Add selected chapters',
        bg='white',
        fg='black',
        font=('Arial', 12),
        command=on_add_select
    )
    add_button.pack(pady=5)

    remove_button = tk.Button(
        middle_frame,
        text='Remove selected chapters',
        bg='white',
        fg='black',
        font=('Arial', 12),
        command=on_remove_select
    )
    remove_button.pack(pady=5)

    add_all_button = tk.Button(
        middle_frame,
        text='Add all chapters',
        bg='white',
        fg='black',
        font=('Arial', 12),
        command=on_add_all
    )
    add_all_button.pack(pady=5)

    # Right listbox and scrollbar
    scrollbar_selected = ttk.Scrollbar(right_frame, orient=tk.VERTICAL)
    scrollbar_selected.pack(side=tk.RIGHT, fill=tk.Y)

    chapters_selected_listbox = tk.Listbox(
        right_frame,
        selectmode=tk.MULTIPLE,
        yscrollcommand=scrollbar_selected.set,
        font=('Arial', 12),
        width=60,
        height=10
    )
    chapters_selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_selected.config(command=chapters_selected_listbox.yview)

    start_convert_button = tk.Button(
        root,
        text='Convert epub',
        command=convert,
        bg='white',
        fg='black',
        font=('Arial', 12)
    )
    start_convert_button.pack(pady=20)

    output_text = tk.Text(root, height=10, width=50, bg="black", fg="white", font=('Arial', 12))
    output_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    output_text.tag_configure("red", foreground="white")
    output_text.insert(tk.END, "Output here....", "red")
    output_text.configure(state='disabled')

    # start main loop
    root.mainloop()


if __name__ == "__main__":
    start_gui()
