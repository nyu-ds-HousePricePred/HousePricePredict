# NYC House Sale Price Prediction

## Contributors and contact
Biqi Lin, bl2379@nyu.edu;
Yuan Ling, yl4452@nyu.edu;
Dajiang Liu, dl3502@nyu.edu;
Runyu Yan, ry787@nyu.edu

## Introduction
The goal of the study is to compare the performance of three machine learning algorithms on predicting housing sales price. The algorithms we chose are K-Nearest-Neighbors (KNN), LASSO (Lasso), and Random Forest (RF) regression. The metric used for measuring performance was root-mean-square error (RMSE).

Data were online-archived real-estate sales data of New York City within the 2016-2017 time period, retrieved from the Department of Finance of the City of New York. There are originally 20 explanatory features, with 84,548 rows of observations.

## Feature list
### Borough:
The name of the borough in which the property is located.
### Neighborhood:
Department of Finance assessors determine the neighborhood name in the course of valuing
properties. The common name of the neighborhood is generally the same as the name
Finance designates. However, there may be slight differences in neighborhood boundary lines
and some sub-neighborhoods may not be included.
### Building Class Category:
This is a field that we are including so that users of the Rolling Sales Files can easily
identify similar properties by broad usage (e.g. One Family Homes) without looking up
individual Building Classes. Files are sorted by Borough, Neighborhood, Building Class
Category, Block and Lot.
### Tax Class at Present:
Every property in the city is assigned to one of four tax classes (Classes 1, 2, 3, and 4),
based on the use of the property.
  
  • Class 1: Includes most residential property of up to three units (such as one-,
two-, and three-family homes and small stores or offices with one or two
attached apartments), vacant land that is zoned for residential use, and most
condominiums that are not more than three stories.
  
  • Class 2: Includes all other property that is primarily residential, such as
cooperatives and condominiums.
  
  • Class 3: Includes property with equipment owned by a gas, telephone or electric
company.
  
  • Class 4: Includes all other properties not included in class 1,2, and 3, such as
offices, factories, warehouses, garage buildings, etc.
Glossary of Terms for Property Sales Files
### Block:
A Tax Block is a sub-division of the borough on which real properties are located.
The Department of Finance uses a Borough-Block-Lot classification to label all real
property in the City. “Whereas” addresses describe the street location of a property, the
block and lot distinguishes one unit of real property from another, such as the different
condominiums in a single building. Also, block and lots are not subject to name changes
based on which side of the parcel the building puts its entrance on.
### Lot:
A Tax Lot is a subdivision of a Tax Block and represents the property unique location.
### Easement:
An easement is a right, such as a right of way, which allows an entity to make limited use of
another’s real property. For example: MTA railroad tracks that run across a portion of another
property.
### Building Class at Present:
The Building Classification is used to describe a property’s constructive use. The first position
of the Building Class is a letter that is used to describe a general class of properties (for
example “A” signifies one-family homes, “O” signifies office buildings. “R” signifies
condominiums). The second position, a number, adds more specific information about the
property’s use or construction style (using our previous examples “A0” is a Cape Cod style
one family home, “O4” is a tower type office building and “R5” is a commercial condominium
unit). The term Building Class used by the Department of Finance is interchangeable with the
term Building Code used by the Department of Buildings. See NYC Building Classifications.
Address: The street address of the property as listed on the Sales File. Coop sales
include the apartment number in the address field.
### Zip Code: 
The property’s postal code
### Residential Units:
The number of residential units at the listed property.
### Commercial Units:
The number of commercial units at the listed property.
### Total Units:
The total number of units at the listed property.
### Land Square Feet:
The land area of the property listed in square feet.
### Gross Square Feet:
The total area of all the floors of a building as measured from the exterior surfaces of the
outside walls of the building, including the land area and space within any building or structure
on the property.
### Year Built:
Year the structure on the property was built.
### Building Class at Time of Sale:
The Building Classification is used to describe a property’s constructive use. The first
position of the Building Class is a letter that is used to describe a general class of
properties (for example “A” signifies one-family homes, “O” signifies office buildings. “R”
signifies condominiums). The second position, a number, adds more specific information
about the property’s use or construction style (using our previous examples “A0” is a Cape
Cod style one family home, “O4” is a tower type office building and “R5” is a commercial
condominium unit). The term Building Class as used by the Department of Finance is
interchangeable with the term Building Code as used by the Department of Buildings.
### Sale Date:
Date the property sold.

## Target variable
### Sales Price:
Price paid for the property.
