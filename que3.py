
prices = [10,14,22,33,44,13,22,55,66,77]
total = 0
print("Supermarket")
print("================")

while True:
    num = int(input("Please select product (1-10) 0 to Quit:"))
    if num == 0:
        break
    if num>=1 and num <=10:
        price = prices[num-1]
        total = total+price
        print("Product:",num,"Price:",price)

print("Total:",total)
pay = int(input("Payment:"))
print("Change:",pay-total)
