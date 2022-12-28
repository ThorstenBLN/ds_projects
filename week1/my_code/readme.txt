project: visual analyzation of the Gapminder Dataset and creation of an animated scatterplot

The gapminder dataset contains population, lifeexpectancy and fertility by country
 for several decades.
The target of the first project was to show the development of the contained countries 
in an animated scatterplot. 

scatterplot:
    - x-axis: lifeexpectancy
    - y-axis: fertility rate
    - datapoint: country
    - datapoint size: population size of the corresponding country.
    - datapoint colour: corresponding continent

libraries used:
- matplotlib/pyplot and seaborn for plotting
- pandas for data
- imageio for creating the animated gif