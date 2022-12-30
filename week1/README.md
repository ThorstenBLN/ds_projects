### visual analyzation of the Gapminder Dataset and creation of an animated scatterplot  

data: The gapminder dataset contains population, lifeexpectancy and fertility by country for several decades.
target: show the development of the contained countries in an animated scatterplot. 

scatterplot:
- x-axis: lifeexpectancy
- y-axis: fertility rate
- datapoint: country
- datapoint size: population size of the corresponding country.
- datapoint colour: corresponding continent

techstack:
- matplotlib/pyplot, seaborn, pandas, imageio
