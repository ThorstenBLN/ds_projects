import customer as cu
import datetime as dt
import csv
from scipy.stats import poisson


class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self, opening_time, trans_m, init_dist, possion_means):       
        '''initialization of class attributes''' 
        self.timestamp = opening_time
        self.trans_m = trans_m # transition matrix for propagating a customer to its next state
        self.init_dist = init_dist # distribution of 1. state of customers
        self.poisson_means = possion_means # distribution for generating new customers
        # a list of Customer objects
        self.customers = [] # list containing all active customers 
        self.minutes = 0
        self.last_id = 0
        # create a csv file to log the customers simulation 
        self.csv_file = open('super.csv', mode='w')
        self.csv_writer = csv.writer(self.csv_file, delimiter=';')


    def __repr__(self):
        return f'\n\ncurrent-time: {self.timestamp}, no of customers: {len(self.customers)}\n'

 
    def print_customers(self):
        """print all customers with the current time and id to CSV log file"""
        for customer in self.customers:
            print(f"""minutes: {self.timestamp} active: {customer.active}, customer_id: {customer.id}, location: {customer.route[-1]},
                route: {customer.route}""")
            self.csv_writer.writerow([self.timestamp, customer.id, customer.route[-1]])
            

    def next_minute(self):
        """increases the time by 1 minute
        """
        self.minutes += 1
        # adding a minute to the timestamp
        self.timestamp += dt.timedelta(0, 0, 0, 0, 1)

        
    def next_move(self):
        """propagates all active customers to the next state.
        """
        for customer in self.customers: # only active customers will be in customers-list
            customer.next_move()

    
    def add_new_customers(self):
        """randomly creates new customers.
        """
        # create new customers with the poisson distribution (by mean of entering customers per minute by hour of sample df)
        no_new_customers = poisson.rvs(self.poisson_means['cus'][self.timestamp.hour], size=1)
        for i in range(no_new_customers[0]):
            # increase the customer_id by 1
            self.last_id += 1
            # create the new customers
            self.customers.append(cu.Customer('cu-' + str(self.last_id), self.trans_m, self.init_dist))


    def remove_exiting_customers(self):
        """removes every customer that is not active any more."""
        # recreates the self.customer list with only active customers
        self.customers = [customer for customer in self.customers if customer.active]
