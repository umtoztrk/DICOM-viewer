
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\ATAKAN\Desktop\gui\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("750x500")
window.configure(bg = "#D36F6F")


canvas = Canvas(
    window,
    bg = "#D36F6F",
    height = 500,
    width = 750,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    375.0,
    250.0,
    image=image_image_1
)

canvas.create_text(
    167.0,
    90.0,
    anchor="nw",
    text="DICOM GÖRÜNTÜLEYİCİ",
    fill="#000000",
    font=("Inter", 36 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=229.0,
    y=335.0,
    width=291.0,
    height=58.0
)
window.resizable(False, False)
window.mainloop()
