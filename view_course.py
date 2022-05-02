# from tkinter import *
# from PIL import ImageTk, Image
# import random

# def create_circle(x, y, r, canvasName): #center coordinates, radius
#     x0 = x - r
#     y0 = y - r
#     x1 = x + r
#     y1 = y + r
#     return canvasName.create_oval(x0, y0, x1, y1)

# def draw_circle(coo, canvasName, root):
#     if len(coo) > 0:
#         new_coo = coo.pop(0)
#         create_circle(new_coo[0], new_coo[1], 10, canvasName)
#         root.after(10000, draw_circle(coo, canvasName,root))

# def view_course(coo):
#     # root, canvas
#     root = Tk()
#     canvas = Canvas(root, height=824, width=447)

#     my_course = ImageTk.PhotoImage(Image.open("course_image/course.PNG"))
#     canvas.create_image(0, 0, anchor=NW, image=my_course)

#     canvas.pack()
#     draw_circle(coo, canvas, root)
#     root.mainloop()

# coo =  [(500.5, 200.5), (300.0, 426.0), (111.0, 233.5)]

# view_course(coo)


# import tkinter as tk
# import random

# def make_segment():
#     return [random.randrange(0, 800) for _ in range(4)]

# def draw_random_lines():
#     print(make_segment(s))
#     canvas.create_line(*make_segment())
#     root.after(100, draw_random_lines)

# root = tk.Tk()
`# canvas = tk.Canvas(root, height=800, width=800)
`# canvas.pack()

# draw_random_lines()

# root.mainloop()


from tkinter import *

x = 10
y = 10
a = 50
b = 50

x_vel = 5
y_vel = 5

def move():

    global x
    global y
    global x_vel
    global y_vel
    if x < 0:
        x_vel = 5
    if x > 350:
        x_vel = -5
    if y < 0:
        y_vel = 5
    if y > 150:
        y_vel = -5
    canvas1.move(circle, x_vel, y_vel)
    coordinates = canvas1.coords(circle)
    x = coordinates[0]
    y = coordinates[1]
    window.after(33, move)

window   = Tk()
window.geometry("1000x1000")

canvas1=Canvas(window, height = 1000, width= 1000)
canvas1.grid (row=0, column=0, sticky=W)
coord = [x, y, a, b ]
circle = canvas1.create_oval(coord, outline="red", fill="red")


# coord = [230, 270, 270, 310]
# rect2 = canvas1.create_rectangle(coord, outline="Blue", fill="Blue")

move()

window.mainloop ()




import tkinter as tk
import random

class Bubble():

    def __init__(self, canvas, x, y, size, color='red'):
        self.canvas = canvas

        self.x = x
        self.y = y

        self.start_x = x
        self.start_y = y

        self.size = size
        self.color = color

        self.circle = canvas.create_oval([x, y, x+size, y+size], outline=color, fill=color)

    def move(self):
        x_vel = random.randint(-5, 5)
        y_vel = -5

        self.canvas.move(self.circle, x_vel, y_vel)
        coordinates = self.canvas.coords(self.circle)

        self.x = coordinates[0]
        self.y = coordinates[1]

        # if outside screen move to start position
        if self.y < -self.size:
            self.x = self.start_x
            self.y = self.start_y
            self.canvas.coords(self.circle, self.x, self.y, self.x + self.size, self.y + self.size)

def move():
    for item in bubbles:
        item.move()

    window.after(33, move)

# --- main ---

start_x = 230
start_y = 270

window = tk.Tk()
window.geometry("1000x1000")

canvas = tk.Canvas(window, height=1000, width=1000)
canvas.grid(row=0, column=0, sticky='w')

bubbles = []
for i in range(5):
    offset = random.randint(10, 20)
    b = Bubble(canvas, start_x+10, start_y-offset, 20, 'red')
    bubbles.append(b)
for i in range(5):
    offset = random.randint(0, 10)
    b = Bubble(canvas, start_x+10, start_y-offset, 20, 'green')
    bubbles.append(b)

coord = [start_x, start_y, start_x+40, start_y+40]
rect = canvas.create_rectangle(coord, outline="Blue", fill="Blue")

move()

window.mainloop ()