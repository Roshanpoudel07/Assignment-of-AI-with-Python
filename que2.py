items = []
while True:
    choice = input("Would you like to\n(1)Add or\n(2)Remove items or\n(3)Quit?:")
    if choice == "1":
        item = input("What will be added?:")
        items.append(item)
    elif choice == "2":
        print("There are", len(items), "items in the list.")
        num = int(input("Which item is deleted?:"))
        if 0 <= num < len(items):
            items.pop(num)
        else:
            print("Incorrect selection.")
    elif choice =="3":
        print("The following items remain in the list:")
        for x in items:
            print(x)
        break
    else:
        print("Incorrect selection.")
