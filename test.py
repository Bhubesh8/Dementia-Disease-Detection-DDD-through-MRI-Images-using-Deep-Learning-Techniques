from datetime import date

present_date = date.today()
formatted_date = present_date.strftime("%d-%m-%Y")
print(type(formatted_date))