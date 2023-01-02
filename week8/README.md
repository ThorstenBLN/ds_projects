### Markov simulation of customer behaviour in a supermarket

#### original data: 
5 files - each one contains 1 weekday of observation data of customers in a supermarket  
columns - timestamp, customer number, state of customer  
customers - can be in 5 states ('checkout', 'dairy', 'drinks', 'fruit', 'spices')  

#### simulation:
Every minute new customers enter the supermarket. These new customers will be individually assigined to an initial state. At the same time all existing active customers propagate to the next state or stay at their current state. Once a customer has reached the checkout state he leaves the supermarket and becomes inactive. For info regarding the used distribtions see files.

#### files:
- customer.py: 
    - contains the customer class
    - handles the customer, its initial state, its transition to the next state, 
    tracks route of the customer through the supermarket, deactivates the customer if it reached checkout
    - initial state: randomly assigned by frequency distribution of 1st states of customers in original data
    - transition to next state: randomly assigned by transition matrix, containing distributions of movements from 
    current state to next state from original data. (In the original data only moving customers are logged. This
    leads to the situation that a customer staying for 10 minutes in one state has only 1 timestamp at 
    the state (when entering). To prevent unrealistic "each minute moves" in the transition matrix, these missing timesteps
    for customers staying in a state have been filled.

- supermarket.py: 
    - contains the supermarket class
    - the supermarket class handles the simulation time, creates new entering customers each minute, 
    propagates active customers to the next state, keeps track of all active customers and removes exiting customers
    - new customer creation: by poisson distibution (1 for each hour) depending on the mean entering customers 
    per minute in the certain hour over the whole week of original data

- supermarket_sim.py: 
    - loads and edits the original data
    - creates the transition matrix, the initial state distribution and the means for the poisson distribution
    for the customer creation
    - contains the eventloop of the supermarket simulation

