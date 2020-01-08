TIME = 2387
BASE = 60

reminder = TIME % BASE
minute = (TIME - reminder) / 60

print("Minute is : %d " % (minute))
print("Reminder is : %d " % (reminder))