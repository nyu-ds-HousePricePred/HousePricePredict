# NYC House Sale Price Prediction

## Contributors and contact
Biqi Lin, bl2379@nyu.edu;
Yuan Ling, yl4452@nyu.edu;
Dajiang Liu, dl3502@nyu.edu;
Runyu Yan, ry787@nyu.edu

## Introduction
The goal of the study is to compare the performance of three machine learning algorithms on predicting housing sales price. The algorithms we chose are K-Nearest-Neighbors (KNN), LASSO (Lasso), and Random Forest (RF) regression. The metric used for measuring performance was root-mean-square error (RMSE).

Data were online-archived real-estate sales data of New York City within the 2016-2017 time period, retrieved from the Department of Finance of the City of New York. There are originally 20 explanatory features, with 84,548 rows of observations.

## Files in the repository
### Figures.zip:
A zip file storing all the figures created in this study.
### HousePricePrediction_Code_Final.ipynb:
A Jupyter Notebook file with the codes of the study in Python 3 environment.
### README.md:
A Github markdown file with the introduction of the study and the instructions of executing the code.
### nyc-rolling-sales.csv:
A comma-separated-values file with the entire original dataset used in the study.
### report_final.pdf:
A pdf file with the final report of the study.

## Running the Jupyter Notebook
(for details, see https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html)

### 1. Launching the Jupyter Notebook App:
The Jupyter Notebook App can be launched by clicking on the Jupyter Notebook icon installed by Anaconda in the start menu (Windows) or by typing "jupyter notebook" in a terminal (cmd on Windows).

This will launch a new browser window (or a new tab) showing the Notebook Dashboard, a sort of control panel that allows (among other things) to select which notebook to open.

When started, the Jupyter Notebook App can access only files within its start-up folder (including any sub-folder). If you store the notebook documents in a subfolder of your user folder no configuration is necessary. Otherwise, you need to choose a folder which will contain all the notebooks and set this as the Jupyter Notebook App start-up folder.

### 2. Change Jupyter Notebook startup folder:

#### For Windows users, follow the steps below:
1) Copy the Jupyter Notebook launcher from the menu to the desktop.

2) Right click on the new launcher and change the Target field, change %USERPROFILE% to the full path of the folder which will contain all the notebooks.

3) Double-click on the Jupyter Notebook desktop launcher (icon shows [IPy]) to start the Jupyter Notebook App, which will open in a new browser window (or tab). Note also that a secondary terminal window (used only for error logging and for shut down) will be also opened. If only the terminal starts, try opening this address with your browser: http://localhost:8888/.

#### For MacOS users, follow the steps below:
1) Click on spotlight, type terminal to open a terminal window.

2) Enter the startup folder by typing cd /some_folder_name.

3) Type jupyter notebook to launch the Jupyter Notebook App (it will appear in a new browser window or tab).

### 3. Execute a notebook:
Download the "HousePricePrediction_Code_Final.ipynb" file to the notebook folder or a sub-folder of it. Launch the Jupyter Notebook App (see previous section). In the Notebook Dashboard navigate to find the notebook: clicking on its name will open it in a new browser tab. Click on the menu Help -> User Interface Tour for an overview of the Jupyter Notebook App user interface. You can run the notebook document step-by-step (one cell a time) by pressing shift + enter. You can run the whole notebook in a single step by clicking on the menu Cell -> Run All. To restart the kernel (i.e. the computational engine), click on the menu Kernel -> Restart. This can be useful to start over a computation from scratch (e.g. variables are deleted, open files are closed, etc...).

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
