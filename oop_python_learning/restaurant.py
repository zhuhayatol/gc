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

class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors=["Red","Blue","Green"]
    
    def list_icecream(self):
        for i in range(len(self.flavors)):
            print(self.flavors[i])

#b=IceCreamStand("xjtu","xi'an")
#b.list_icecream()