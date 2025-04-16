class Restaurant:
    def __init__(self,restaurant_name,cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        
    def describe_restaurant(self):
        print(f"{self.restaurant_name} serves {self.cuisine_type} cuisine.")
    
    def open_restaurant(self):
        print(f"{self.restaurant_name} is open!")

restaurant=Restaurant("xjtu","Sichuan")
restaurant1=Restaurant("sjtu","Shanghai")
restaurant2=Restaurant("thu","Beijing")
re1=Restaurant("lhl","anhui")
print(restaurant.restaurant_name,restaurant.cuisine_type)


restaurant.describe_restaurant()
restaurant1.describe_restaurant()
restaurant2.describe_restaurant()
re1.describe_restaurant()
