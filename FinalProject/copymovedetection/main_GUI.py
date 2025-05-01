from tkinter import Frame, Label, Tk, BOTH, Text, END
from tkinter.ttk import Button, Style
import tkinter.filedialog
import tkinter.messagebox as mbox
from PIL import Image, ImageTk
import os
import shutil
import CopyMoveDetection

class aFrame(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.imageName = ""
        self.imagePath = ""
        self.initUI()

    def initUI(self):
        self.parent.title("Image Copy-Move Detection")
        self.style = Style().configure("TFrame", background="#333")
        self.pack(fill=BOTH, expand=1)

        openButton = Button(self, text="Open File", command=self.onFilePicker)
        openButton.place(x=10, y=10)

        detectButton = Button(self, text="Detect", command=self.onDetect)
        detectButton.place(x=10, y=40)

        self.textBoxFile = Text(self, state='disabled', width=80, height=1)
        self.textBoxFile.place(x=90, y=10)

        self.textBoxLog = Text(self, state='disabled', width=80, height=3)
        self.textBoxLog.place(x=90, y=40)

        self.labelLeft = Label(self)
        self.labelLeft.place(x=5, y=100)

        self.labelRight = Label(self)
        self.labelRight.place(x=525, y=100)

        self.centerWindow()

    def centerWindow(self):
        w = 1045
        h = 620
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def onFilePicker(self):
        ftypes = [('Image Files', '*.png *.jpg *.jpeg'), ('All files', '*')]
        dlg = tkinter.filedialog.Open(self, initialdir='.', filetypes=ftypes)
        selectedFile = dlg.show()

        if selectedFile:
            self.imagePath = os.path.dirname(selectedFile) + "/"
            self.imageName = os.path.basename(selectedFile)

            self.textBoxFile.config(state='normal')
            self.textBoxFile.delete('1.0', END)
            self.textBoxFile.insert(END, selectedFile)
            self.textBoxFile.config(state='disabled')

            newImageLeft = Image.open(selectedFile)
            imageLeftLabel = ImageTk.PhotoImage(newImageLeft)
            self.labelLeft.config(image=imageLeftLabel)
            self.labelLeft.image = imageLeftLabel

            self.labelRight.config(image='')

    def onDetect(self):
        if not self.imageName:
            mbox.showerror("Error", 'No image selected\nSelect an image by clicking "Open File"')
            return

        self.textBoxLog.config(state='normal')
        self.textBoxLog.insert(END, "Detecting: " + self.imageName + "\n")
        self.textBoxLog.see(END)
        self.textBoxLog.config(state='disabled')

        test_images_path = "copy_move_detection/test_images/"
        results_path = "copy_move_detection/results/"
        os.makedirs(test_images_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        input_full_path = os.path.join(self.imagePath, self.imageName)
        copied_path = os.path.join(test_images_path, self.imageName)
        shutil.copy(input_full_path, copied_path)

        imageResultPath = CopyMoveDetection.detect(test_images_path, self.imageName, results_path, blockSize=32)
        newImageRight = Image.open(imageResultPath)
        imageRightLabel = ImageTk.PhotoImage(newImageRight)
        self.labelRight.config(image=imageRightLabel)
        self.labelRight.image = imageRightLabel

        self.textBoxLog.config(state='normal')
        self.textBoxLog.insert(END, "Detection complete.\n")
        self.textBoxLog.see(END)
        self.textBoxLog.config(state='disabled')

if __name__ == '__main__':
    root = Tk()
    app = aFrame(root)
    root.mainloop()