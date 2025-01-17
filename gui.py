import tkinter as tk
from tkinter import filedialog, ttk
from audiblez import main, get_kokoro, get_voice_list
import sys
import threading


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


def start_gui():
    root = tk.Tk()
    root.title('Audiblez')
    root.geometry('600x400')
    root.resizable(True, True)

    def select_file():
        file_path = filedialog.askopenfilename(
            title='Select an epub file',
            filetypes=[('epub files', '*.epub')]
        )
        if file_path:
            file_label.config(text=file_path)
            kokoro = get_kokoro()        
            output_text.configure(state='normal')
            output_text.delete(1.0, tk.END)
            output_text.configure(state='disabled')
            # Redirect stdout to Text widget
            sys.stdout = TextRedirector(output_text)
            voice = voice_combo.get()
            threading.Thread(target=main, args=(kokoro, file_path, "en-gb", voice, False, 1.0)).start()
    
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

    output_text = tk.Text(root, height=10, width=50, bg="black", fg="white", font=('Arial', 12))
    output_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    output_text.tag_configure("red", foreground="white")
    output_text.insert(tk.END, "Output here....", "red")
    output_text.configure(state='disabled')
    
    # add a combo box with voice options
    voice_label = tk.Label(root, text="Select Voice:", font=('Arial', 12))
    voice_label.pack(pady=5)

    # add a combo box with voice options
    voices = get_voice_list()
    voice_combo = ttk.Combobox(
        root, 
        values=voices,
        state="readonly",
        font=('Arial', 12)
    )
    voice_combo.set(voices[0])  # Set default selection
    voice_combo.pack(pady=10)

    # start main loop
    root.mainloop()


if __name__ == "__main__":
    start_gui()
else:
    print(f"__name__ is {__name__}")
