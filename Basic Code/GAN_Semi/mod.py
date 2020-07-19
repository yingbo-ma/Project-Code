TIME = 540
BASE = 60

reminder = TIME % BASE
minute = (TIME - reminder) / 60

print("Minute is : %d " % (minute))
print("Second is : %d " % (reminder))