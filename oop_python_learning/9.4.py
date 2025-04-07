class Restaurant:
    def __init__(self,restaurant_name,cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0
        
    def describe_restaurant(self):
        print(f"{self.restaurant_name} serves {self.cuisine_type} cuisine.")
    
    def open_restaurant(self):
        print(f"{self.restaurant_name} is open!")

    def set_number_served(self,people):
        self.number_served=people 
    
    def increment_number_served(self,increase):
        self.number_served+=increase

a = Restaurant("xjtu","xi'an")
a.set_number_served(3000)
print(a.number_served)
a.increment_number_served(100)
print(a.number_served)