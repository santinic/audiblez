import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from engine import main, get_kokoro, get_voice_list
import sys
import threading
import onnxruntime as ort


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


def get_language_from_voice(voice):
    if voice.startswith("a"):
        return "en-us"
    elif voice.startswith("b"):
        return "en-gb"
    else:
        print("Voice not recognized.")
        exit(1)


def start_gui():
    root = tk.Tk()
    root.title('Audiblez')
    root.geometry('900x600')
    root.resizable(True, True)

    # ui element variables
    pick_chapters_var = tk.BooleanVar()

    def select_file():
        file_path = filedialog.askopenfilename(
            title='Select an epub file',
            filetypes=[('epub files', '*.epub')]
        )
        if file_path:
            file_label.config(text=file_path)
    
    def convert():    
        def enable_controls():
            speed_scale.configure(state='normal')
            providers_combo.configure(state='normal')
            voice_combo.configure(state='normal')
        
        def run_conversion():
            try:
                main(kokoro, file_path, language, voice, pick_chapters, speed, [provider])
            finally:
                # Ensure controls are re-enabled even if an error occurs
                root.after(0, enable_controls)

        if file_label.cget("text"):
            kokoro = get_kokoro()        
            output_text.configure(state='normal')
            output_text.delete(1.0, tk.END)
            output_text.configure(state='disabled')
            # Redirect stdout to Text widget
            sys.stdout = TextRedirector(output_text)
            file_path = file_label.cget("text")
            voice = voice_combo.get()
            provider = providers_combo.get()
            speed = speed_scale.get()
            pick_chapters = pick_chapters_var.get()
            language = get_language_from_voice(voice)
            speed_scale.configure(state='disabled')
            providers_combo.configure(state='disabled')
            voice_combo.configure(state='disabled')
            threading.Thread(target=run_conversion).start()
            # when this thread finishes, re-enable the buttons

        else:
            warning = "Please select an epub file first."
            print(warning)
            # create a warning message box to say this
            messagebox.showwarning("Warning", warning)

    
    file_button = tk.Button(
        root,
        text='Select epub file',
        command=select_file,
        bg='white',
        fg='black',
        font=('Arial', 12)
    )
    file_button.pack(pady=20)

    file_label = tk.Label(root, text="")
    file_label.pack(pady=5)

    start_convert_button = tk.Button(
        root,
        text='Convert epub',
        command=convert,
        bg='white',
        fg='black',
        font=('Arial', 12)
    )
    start_convert_button.pack(pady=20)

    # add a check box to pick or not pick chapters
    pick_chapters_check = tk.Checkbutton(
        root,
        text="Pick chapters",
        variable=pick_chapters_var,
        font=('Arial', 12)
    )

    pick_chapters_check.configure(state='disabled')
    pick_chapters_check.pack(pady=5)

    voice_frame = tk.Frame(root)
    voice_frame.pack(pady=5, padx=5)

    # add a scale to set speed
    speed_label = tk.Label(voice_frame, text="Set speed:", font=('Arial', 12))
    speed_label.pack(side=tk.LEFT, pady=5, padx=5)

    speed_scale = tk.Scale(
        voice_frame,
        from_=0.5,
        to=2.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        font=('Arial', 12)
    )
    speed_scale.set(1.0)
    speed_scale.pack(side=tk.LEFT, pady=5, padx=5)

    # add a combo box with ONNX providers
    available_providers = ort.get_available_providers()
    default_provider = [p for p in available_providers if "CPU" in p][0]
    providers_label = tk.Label(voice_frame, text="Select ONNX providers:", font=('Arial', 12))
    providers_label.pack(side=tk.LEFT, pady=5, padx=5)

    providers_combo = ttk.Combobox(
        voice_frame,
        values=available_providers,
        state="readonly",
        font=('Arial', 12)
    )
    providers_combo.set(default_provider)  # Set default selection
    providers_combo.pack(side=tk.LEFT, pady=5, padx=5)
    
    # add a combo box with voice options
    voice_label = tk.Label(voice_frame, text="Select Voice:", font=('Arial', 12))
    voice_label.pack(side=tk.LEFT, pady=5, padx=5)

    # add a combo box with voice options
    voices = get_voice_list()
    voice_combo = ttk.Combobox(
        voice_frame,
        values=voices,
        state="readonly",
        font=('Arial', 12)
    )
    voice_combo.set(voices[0])  # Set default selection
    voice_combo.pack(side=tk.LEFT, pady=10, padx=5)

    output_text = tk.Text(root, height=10, width=50, bg="black", fg="white", font=('Arial', 12))
    output_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    output_text.tag_configure("red", foreground="white")
    output_text.insert(tk.END, "Output here....", "red")
    output_text.configure(state='disabled')

    # start main loop
    root.mainloop()


if __name__ == "__main__":
    start_gui()
else:
    print(f"__name__ is {__name__}")
