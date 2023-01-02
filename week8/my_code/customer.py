import numpy as np

class Customer():
    
    def __init__(self, id, transaction_matrix, initial_distribution):
        '''creates the customer object and initialises the attributes'''
        # create properties
        self.id = id
        self.transaction_matrix = transaction_matrix # transition matrix to the next state
        self.initial_distribution = initial_distribution # distribution of the 1. state
        # instantiate a random default Generator (enables to pick from multi-D-List)
        self.rng = np.random.default_rng()
        # vector of current position. Initially it is None. 
        # later a binary vector with 1 at current state all other states have 0
        self.current_vector = None 
        # location list is in same order as the numpy arrays and transaction matrix
        self.locations = ['checkout', 'dairy', 'drinks', 'fruit', 'spices'] 
        # route the customer takes in the supermarket
        self.route = []
        # defines if customer has reached checkout
        self.active = True


    def __repr__(self):
        return f"""class: Customer, current_vector: {self.current_vector}: {self.get_location_from_vector(self.current_vector == 1)},
                path: {self.route}"""

    
    def next_move(self):
        '''new customer makes first move. Existing customer makes the next move '''
        if self.current_vector is None:
            # new customer --> 1st step: creates the first array (which indicates the location)
            self.current_vector = np.array(self.rng.choice([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], 
                [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], size=1, p=self.initial_distribution)[0]) 
        else:
            # existing customer --> next step: get the vector of the next location
            self.current_vector = np.array(self.rng.choice([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], 
                    [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], size=1, p=np.dot(self.current_vector, self.transaction_matrix))[0])
        # append the new location to route
        self.route.append(self.get_location_from_vector(self.current_vector)) 
        if self.get_location_from_vector(self.current_vector) == 'checkout':
            self.active = False
        

    def get_location_from_vector(self, vector):
        '''translating the np.array into the location string'''
        return self.locations[int(np.where(vector == 1)[0])]
