
DS7331 - Lab 1<br>
Submitted by: Shravan Reddy, Samira Zarandioon, Jaime Villanueva
<br><br>
<center><h1> Forest Cover Type Analysis </h1></center>



<a id='toc'></a>
# Table of Contents

1. [Introduction](#Introduction)<br>
2. [Business Understanding](#Business_Understanding)<br>
3. [Data Description](#Data_Description)
4. [Data Quality](#Data_Quality)<br>
5. [Basic Statistics and Visualizations](#Basic_Stats)<br>
6. [Attributes of Interest](#Attribute_Interest)<br>
7. [Attribute - Attribute Relationships](#Attribute_Attribute)<br>
8. [Attribute - Response Relationships](#Attribute_Response)<br>
9. [Additional Features](#Additional_Features)<br>
10. [Exceptional Work](#Exceptional_Work)

<a id='Introduction'></a>
# Introduction

The Roosevelt Natonal Forest is located about 100 miles northwest of Denver, Colorado, and is an area of more than 800,000 acres of land. The areas of interest in this forest for this analysis are the four wilderness areas, Rawah, Comanche Peak, Neota, and Cache la Poudre. A wilderness area is an official legal designation created by the Wilderness Preservation Act in 1964. This act created the Wilderness Preservation System and sets aside land areas in the United States to be managed and maintained in its natural wild state. This management is administered by four different government agencies: the National Park Service, the U.S. Forest Service (USFS), U.S. Fish and Wildlife Service, and the Bureau of Land Management.

![Wilderness Areas](7331lab1image1.png)
images taken from : https://www.wilderness.net/NWPS/maps

[top](#toc)
<a id='Business_Understanding'></a>
# Business Understanding

Being able to accurately catalog the natural resources of an area is important to land management agencies. In order to maintain the natural state of the forest, the natural resource managers are responsible for developing ecosystem management strategies. This process requires the collection of information from large areas of land in order to properly inventory an ecosystem, however the actual collection of such information can be time and cost prohibitive. Good predictive modeling can serve as an alternate method for creating these necessary inventories.

One of the most basic pieces of information that is collected from wildlife areas is the type of trees that are present. If a predictive model could take other attributes of the land that are either known or are easier to collect, and then use the information from these attributes to accurately predict the type of trees that would be found under those conditions, this has the potential to have a big cost and time saving for the federal agencies managing the area. The data in this analysis was derived from data originally obtained from US Geological Survey (USGS) and USFS data.  The data comes from the aforementioned wilderness areas so they should have minimal human interactions, so we can be more confident that the current forest cover type is more a result of natural ecological processes rather than forest management practices. The data is a combination of information about the terrain with mapping information gathered by agencies using modeling software. Also the type of trees has been collected for these areas, so it is possible to develop a model based on the attributes to predict what type of land cover will be there. Then this can be compared against the actual cover types to get a sense of the accuracy of the model. The model may also be able to weed out any data that is not helpful in the determination of the cover type which could potentially have a cost saving as well. Since the there are several tree types, and the data is a collection of both numerical and categorical, potential models to use would be Linear Discriminant Analysis (LDA), Multi-nomial Regression, or some other classification algorithm such as Artificial Neural Network (ANN).


[top](#toc)
# Data Understanding


<a id='Data_Description'></a>
## Data Description

This data was taken from the UCI Machine Learning Archive: https://archive.ics.uci.edu/ml/datasets/covertype<br>

Data can be downloaded from here: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/<br>

References for data information:<br>
https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info<br>
http://web.cs.ucdavis.edu/~matloff/matloff/public_html/132/Data/ForestCover/BlackardDean.pdf

The four wilderness areas that the data was taken from vary greatly in size. The Rawah wilderness area is 73,213 acres. Comanche Peak is 67,680 acres. Neota is 9647 acres, and Cache la Poudre is 9433 acres. Each record is defined by a 30 x 30 meter cell from a computer model used by the USGS. This cell is directly defined by a digital elevation model (DEM) and is the source for the elevation attribute. All other attributes are based on this 30 x 30 meter cell. There are a total of 581,012 records each representing a cell.

There are ten numeric attributes which measure position relative to various features for each cell, as well as the amount of light at three different times of day during the summer solstice. The light measure is an index estimated by computer models. Also recorded is the amount of slope in each cell.

There are three different categorical attributes listed as well as two that are hidden. One of the categorical attributes is the cover type which is the variable that is being predicted. There are seven types of trees listed each being represented by an integer. The other categorical features are wilderness area and soil type both of which come dummy encoded in the data. There are four wilderness areas and forty soil types. The hidden categorical data are the climate zone and geologic zone which can be deduced from the USFS Ecological Landtype Unit (ELU) code listed for each soil type. The details for the attributes are listed below.

Attribute information:  
  
  Name  | Data Type | Measurement | Description 
  :------------- | :------------- | ------------- | -------------
Elevation  | quantitative  | meters  | Elevation in meters
Aspect  | quantitative  | azimuth  | Aspect in degrees azimuth
Slope  | quantitative  | degrees  | Slope in degrees
Horizontal_Distance_To_Hydrology  | quantitative  | meters  | Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology  | quantitative  | meters  | Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways  | quantitative  | meters  | Horz Dist to nearest roadway
Hillshade_9am  | quantitative  | 0 to 255 index  | Hillshade index at 9am, summer solstice
Hillshade_Noon  | quantitative  | 0 to 255 index  | Hillshade index at noon, summer soltice
Hillshade_3pm  | quantitative  | 0 to 255 index  | Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points  | quantitative  | meters  | Horz Dist to nearest wildfire ignition points
Rawah Wilderness Area  | qualitative  | 0 (absence) or 1 (presence)  | Wilderness area designation
Neota Wilderness Area  | qualitative  | 0 (absence) or 1 (presence)  | Wilderness area designation
Comanche Peak Wilderness Area  | qualitative  | 0 (absence) or 1 (presence)  | Wilderness area designation
Cache la Poudre Wilderness Area  | qualitative  | 0 (absence) or 1 (presence)  | Wilderness area designation
Soil_Type (40 binary columns)  | qualitative  | 0 (absence) or 1 (presence)  | Soil Type designation
Cover_Type (7 types)  | integer  | 1 to 7  | Forest Cover Type designation

Code Designations:
    
Soil Types:             1 to 40 : based on the USFS Ecological
                        Landtype Units (ELUs) for this study area:

  Study Code  | USFS ELU Code | Description 
  ------------- |-------------| ------------- 
1  | 2702  | Cathedral family - Rock outcrop complex, extremely stony.
	 2  | 2703  | Vanet - Ratake families complex, very stony.
	 3  | 2704  | Haploborolis - Rock outcrop complex, rubbly.
	 4  | 2705  | Ratake family - Rock outcrop complex, rubbly.
	 5  | 2706  | Vanet family - Rock outcrop complex complex, rubbly.
	 6  | 2717  | Vanet - Wetmore families - Rock outcrop complex, stony.
	 7  | 3501  | Gothic family.
	 8  | 3502  | Supervisor - Limber families complex.
	 9  | 4201  | Troutville family, very stony.
	10  | 4703  | Bullwark - Catamount families - Rock outcrop complex, rubbly.
	11  | 4704  | Bullwark - Catamount families - Rock land complex, rubbly.
	12  | 4744  | Legault family - Rock land complex, stony.
	13  | 4758  | Catamount family - Rock land - Bullwark family complex, rubbly.
	14  | 5101  | Pachic Argiborolis - Aquolis complex.
	15  | 5151  | unspecified in the USFS Soil and ELU Survey.
	16  | 6101  | Cryaquolis - Cryoborolis complex.
	17  | 6102  | Gateview family - Cryaquolis complex.
	18  | 6731  | Rogert family, very stony.
	19  | 7101  | Typic Cryaquolis - Borohemists complex.
	20  | 7102  | Typic Cryaquepts - Typic Cryaquolls complex.
	21  | 7103  | Typic Cryaquolls - Leighcan family, till substratum complex.
	22  | 7201  | Leighcan family, till substratum, extremely bouldery.
	23  | 7202  | Leighcan family, till substratum - Typic Cryaquolls complex.
	24  | 7700  | Leighcan family, extremely stony.
	25  | 7701  | Leighcan family, warm, extremely stony.
	26  | 7702  | Granile - Catamount families complex, very stony.
	27  | 7709  | Leighcan family, warm - Rock outcrop complex, extremely stony.
	28  | 7710  | Leighcan family - Rock outcrop complex, extremely stony.
	29  | 7745  | Como - Legault families complex, extremely stony.
	30  | 7746  | Como family - Rock land - Legault family complex, extremely stony.
	31  | 7755  | Leighcan - Catamount families complex, extremely stony.
	32  | 7756  | Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
	33  | 7757  | Leighcan - Catamount families - Rock outcrop complex, extremely stony.
	34  | 7790  | Cryorthents - Rock land complex, extremely stony.
	35  | 8703  | Cryumbrepts - Rock outcrop - Cryaquepts complex.
	36  | 8707  | Bross family - Rock land - Cryumbrepts complex, extremely stony.
	37  | 8708  | Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
	38  | 8771  | Leighcan - Moran families - Cryaquolls complex, extremely stony.
	39  | 8772  | Moran family - Cryorthents - Leighcan family complex, extremely stony.
	40  | 8776  | Moran family - Cryorthents - Rock land complex, extremely stony

Forest Cover Type Classes:

Measurement | Description 
  ------------- | ------------- 
1  | Spruce/Fir  
2  | Lodgepole Pine  
3  | Ponderosa Pine  
4  | Cottonwood/Willow
5  | Aspen  
6  | Douglas-fir  
7  | Krummholz  

Note:

|First digit:  climatic zone|Second digit:  geologic zones|
|--|--|
|<table> <tr><th>First digit</th><th>climatic zone</th></tr><tr><td>1</td><td>lower montane dry</td></tr><tr><td>2</td><td>lower montane</td></tr><tr><td>3</td><td>montane dry</td></tr><tr><td>4</td><td>montane</td></tr><tr><td>5</td><td>montane dry and montane</td></tr><tr><td>6</td><td>montane and subalpine</td></tr><tr><td>7</td><td>subalpine</td></tr><tr><td>8</td><td>alpine</td></tr> </table>| <table> <tr><th>Second digit</th><th>geologic zones</th></tr><tr><td>1</td><td>alluvium</td></tr><tr><td>2</td><td>glacial</td></tr><tr><td>3</td><td>shale</td></tr><tr><td>4</td><td>sandstone</td></tr><tr><td>5</td><td>mixed sedimentary</td></tr><tr><td>6</td><td>unspecified in the USFS ELU Survey</td></tr><tr><td>7</td><td>igneous and metamorphic</td></tr><tr><td>8</td><td>volcanic</td></tr> </table>

[top](#toc)
<a id='Data_Quality'></a>
## Data Quality

Because the data came from the a machine learning repository, it is already clean, but there are not any headings for the columns. It is advertised as having no missing data, and this is verified with code. Also the categorical predictors are already dummy coded. For certain analysis techniques, this is nice, but many of the visualizations planned seemed easier without the dummy coding. Therefore after the data is read and labeled, the categorical attributes were collapsed. Also the two hidden categorical attributes were added.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
```


```python
#Read in the .data file into a pandas dataframe
df = pd.read_csv('data/covtype.data', header = None)
```

#### Check for missing data and duplicates


```python
# to check if there is any missing value inf df
df.isnull().values.any()
# there is no missing value
```




    False




```python
# to get number of duplicated rows in df
len(df[df.duplicated()])
# there is no duplicated row
```




    0



#### Label the columns


```python
#Create Names for the columns based on covtypeinfo.txt
quantitative = ['Elevation', 'Aspect', 'Slope', 'hDistance_to_Hydrology', 'vDistance_to_Hydrology', \
                    'hDistance_to_Roads', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'hDistance_to_Fire Points']
wilderness_area = ['Rawah', 'Neota', 'Comanche_Peak', 'Cache_la_Poudre'] 

soil_type = ['Soil Type ' + str(i) for i in range(1,41)]

cover_type = ['Cover_Type']

#Assign names to columns 
df.columns = quantitative + wilderness_area + soil_type  + cover_type
```

#### Undoing the dummy coding


```python
#Preparing the dataframe with no dummy coding

#Create separate df preparing for reversing dummy coding for categorical variables
df_wilderness_area = df.iloc[:,10:14]
df_soil_type = df.iloc[:,14:54]


#Reverse dummy coding for wilderness area and soil type
df['Wilderness_Area'] = pd.Series(df_wilderness_area.columns[np.where(df_wilderness_area !=0)[1]])
df['Soil_Type'] = pd.Series(df_soil_type.columns[np.where(df_soil_type !=0)[1]])
```

#### Replacing cover type integer values with names


```python
#Map Cover Type Names into new column
cover_type_map = {1:"Spruce/Fir", 2:"Lodgepole Pine", 3:"Ponderosa Pine", 4:"Cottonwood/Willow", 5:"Aspen", \
                 6:"Douglas-fir", 7:"Krummholz"}
df['Cover Type Names'] = df['Cover_Type'].map(cover_type_map)
```

#### Creating climatic and geologic attributes from hidden categorical data


```python
#Map ELU codes into new column which will be used to generate columns for climatic and geological zones
elu_map = {"Soil Type 1": "2702", "Soil Type 2":"2703", "Soil Type 3": "2704", "Soil Type 4":"2705", "Soil Type 5":"2706", \
           "Soil Type 6":"2717", "Soil Type 7":"3501", "Soil Type 8":"3502", "Soil Type 9":"4201", "Soil Type 10": "4703", \
           "Soil Type 11":"4704", "Soil Type 12":"4744", "Soil Type 13":"4758", "Soil Type 14":"5101", "Soil Type 15": "5151", \
           "Soil Type 16":"6101", "Soil Type 17":"6102", "Soil Type 18":"6731", "Soil Type 19":"7101", "Soil Type 20": "7102", \
           "Soil Type 21":"7103", "Soil Type 22":"7201", "Soil Type 23":"7202", "Soil Type 24":"7700", "Soil Type 25": "7701", \
           "Soil Type 26":"7702", "Soil Type 27":"7709", "Soil Type 28":"7710", "Soil Type 29":"7745", "Soil Type 30": "7746", \
           "Soil Type 31":"7755", "Soil Type 32":"7756", "Soil Type 33":"7757", "Soil Type 34":"7790", "Soil Type 35": "8703", \
           "Soil Type 36":"8707", "Soil Type 37":"8708", "Soil Type 38":"8771", "Soil Type 39":"8772", "Soil Type 40": "8776" }          
df['ELU Codes'] = df['Soil_Type'].map(elu_map)

#Create Climatic Zone column and map values into it
climatic_map = {"1":"Lower Montane Dry", "2":"Lower Montane", "3":"Montane Dry", "4":"Montane", \
            "5":"Montane Dry and Montane", "6":"Montane and Subalpine", "7":"Subalpine", "8":"Alpine"}

climatic_zone = []
for record in df['ELU Codes']:      #creates list from first digits of the ELU code
    climatic_zone.append(record[0])
df['Climatic_Zone'] = climatic_zone #column is filled with first digits of ELU code
df['Climatic_Zone'] = df['Climatic_Zone'].map(climatic_map) #map first digits to description

#Create Geologic Zone column and map values into it
geologic_map = {"1":"Alluvium", "2":"Glacial", "3":"Shale", "4":"Sandstone", "5":"Mixed Sedimentary", "6":"Unspecified", \
                "7":"Igneous and Metamorphic", "8":"Volcanic"} 

geologic_zone = []
for record in df['ELU Codes']:      #creates list from second digits of the ELU code
    geologic_zone.append(record[1])
df['Geologic_Zone'] = geologic_zone #column is filled with first digits of ELU code
df['Geologic_Zone'] = df['Geologic_Zone'].map(geologic_map) #map first digits to description
```

#### Dropping columns not needed, defining categorical types, and re-ordering columns


```python
#Drop dummy coded columns for wilderness area and soil type and the cover type column with integer values
df = df.drop(wilderness_area, axis=1)
df = df.drop(soil_type, axis=1)
df = df.drop('Cover_Type', axis=1)
df = df.drop('ELU Codes', axis=1)

#Make categorical as category type
df['Wilderness_Area'] = df['Wilderness_Area'].astype('category')
df['Soil_Type'] = df['Soil_Type'].astype('category')
df['Cover Type Names'] = df['Cover Type Names'].astype('category')
df['Climatic_Zone'] = df['Climatic_Zone'].astype('category')
df['Geologic_Zone'] = df['Geologic_Zone'].astype('category')

#Make the response variable last and rename to Cover Type
df['Cover_Type'] = df['Cover Type Names']
df = df.drop('Cover Type Names', axis=1)
```

#### The final prepared dataset for analysis


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 581012 entries, 0 to 581011
    Data columns (total 15 columns):
    Elevation                   581012 non-null int64
    Aspect                      581012 non-null int64
    Slope                       581012 non-null int64
    hDistance_to_Hydrology      581012 non-null int64
    vDistance_to_Hydrology      581012 non-null int64
    hDistance_to_Roads          581012 non-null int64
    Hillshade_9am               581012 non-null int64
    Hillshade_Noon              581012 non-null int64
    Hillshade_3pm               581012 non-null int64
    hDistance_to_Fire Points    581012 non-null int64
    Wilderness_Area             581012 non-null category
    Soil_Type                   581012 non-null category
    Climatic_Zone               581012 non-null category
    Geologic_Zone               581012 non-null category
    Cover_Type                  581012 non-null category
    dtypes: category(5), int64(10)
    memory usage: 47.1 MB



```python
df.head().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elevation</th>
      <td>2596</td>
      <td>2590</td>
      <td>2804</td>
      <td>2785</td>
      <td>2595</td>
    </tr>
    <tr>
      <th>Aspect</th>
      <td>51</td>
      <td>56</td>
      <td>139</td>
      <td>155</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Slope</th>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <th>hDistance_to_Hydrology</th>
      <td>258</td>
      <td>212</td>
      <td>268</td>
      <td>242</td>
      <td>153</td>
    </tr>
    <tr>
      <th>vDistance_to_Hydrology</th>
      <td>0</td>
      <td>-6</td>
      <td>65</td>
      <td>118</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>hDistance_to_Roads</th>
      <td>510</td>
      <td>390</td>
      <td>3180</td>
      <td>3090</td>
      <td>391</td>
    </tr>
    <tr>
      <th>Hillshade_9am</th>
      <td>221</td>
      <td>220</td>
      <td>234</td>
      <td>238</td>
      <td>220</td>
    </tr>
    <tr>
      <th>Hillshade_Noon</th>
      <td>232</td>
      <td>235</td>
      <td>238</td>
      <td>238</td>
      <td>234</td>
    </tr>
    <tr>
      <th>Hillshade_3pm</th>
      <td>148</td>
      <td>151</td>
      <td>135</td>
      <td>122</td>
      <td>150</td>
    </tr>
    <tr>
      <th>hDistance_to_Fire Points</th>
      <td>6279</td>
      <td>6225</td>
      <td>6121</td>
      <td>6211</td>
      <td>6172</td>
    </tr>
    <tr>
      <th>Wilderness_Area</th>
      <td>Rawah</td>
      <td>Rawah</td>
      <td>Rawah</td>
      <td>Rawah</td>
      <td>Rawah</td>
    </tr>
    <tr>
      <th>Soil_Type</th>
      <td>Soil Type 29</td>
      <td>Soil Type 29</td>
      <td>Soil Type 12</td>
      <td>Soil Type 30</td>
      <td>Soil Type 29</td>
    </tr>
    <tr>
      <th>Climatic_Zone</th>
      <td>Subalpine</td>
      <td>Subalpine</td>
      <td>Montane</td>
      <td>Subalpine</td>
      <td>Subalpine</td>
    </tr>
    <tr>
      <th>Geologic_Zone</th>
      <td>Igneous and Metamorphic</td>
      <td>Igneous and Metamorphic</td>
      <td>Igneous and Metamorphic</td>
      <td>Igneous and Metamorphic</td>
      <td>Igneous and Metamorphic</td>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <td>Aspen</td>
      <td>Aspen</td>
      <td>Lodgepole Pine</td>
      <td>Lodgepole Pine</td>
      <td>Aspen</td>
    </tr>
  </tbody>
</table>
</div>



<br><br>
Before doing any analysis we check again to make sure there is no missing or duplicate data after all our data manipulation. If there were either of those, that record would have to be investigated to see if it should be kept or deleted.


```python
# to check if there is any missing value inf df
df.isnull().values.any()
# there is no missing value
```




    False




```python
# to get number of duplicated rows in df
len(df[df.duplicated()])
# there is no duplicated row
```




    0



#### Outliers

The outliers are probably better determined by looking at quick box plots and comparing to basic statistics, but we run some arbitrary numbers first to just get a feel for the data. We used nine times the standard deviation as the benchmark for outlier. Of course, whether this is a big number and whether it is an outlier or not depends on the spread. Do it this way returned forty-five outliers which considering the size of the data set would seem pretty good. But we will a more visual approach as well.


```python
numeric_df = df[quantitative]
outliers = numeric_df[(np.abs( numeric_df-numeric_df.mean())> (9*numeric_df.std())).any(axis=1)]
outliers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>hDistance_to_Hydrology</th>
      <th>vDistance_to_Hydrology</th>
      <th>hDistance_to_Roads</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>hDistance_to_Fire Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>220084</th>
      <td>2954</td>
      <td>290</td>
      <td>31</td>
      <td>845</td>
      <td>581</td>
      <td>939</td>
      <td>121</td>
      <td>219</td>
      <td>230</td>
      <td>2438</td>
    </tr>
    <tr>
      <th>220445</th>
      <td>2960</td>
      <td>286</td>
      <td>27</td>
      <td>854</td>
      <td>588</td>
      <td>953</td>
      <td>134</td>
      <td>226</td>
      <td>227</td>
      <td>2408</td>
    </tr>
    <tr>
      <th>220812</th>
      <td>2963</td>
      <td>279</td>
      <td>23</td>
      <td>864</td>
      <td>589</td>
      <td>960</td>
      <td>152</td>
      <td>236</td>
      <td>221</td>
      <td>2379</td>
    </tr>
    <tr>
      <th>221187</th>
      <td>2949</td>
      <td>288</td>
      <td>28</td>
      <td>847</td>
      <td>573</td>
      <td>940</td>
      <td>132</td>
      <td>225</td>
      <td>227</td>
      <td>2343</td>
    </tr>
    <tr>
      <th>221188</th>
      <td>2963</td>
      <td>283</td>
      <td>19</td>
      <td>874</td>
      <td>588</td>
      <td>968</td>
      <td>164</td>
      <td>237</td>
      <td>212</td>
      <td>2350</td>
    </tr>
    <tr>
      <th>221567</th>
      <td>2955</td>
      <td>293</td>
      <td>25</td>
      <td>859</td>
      <td>583</td>
      <td>949</td>
      <td>143</td>
      <td>225</td>
      <td>219</td>
      <td>2314</td>
    </tr>
    <tr>
      <th>221956</th>
      <td>2949</td>
      <td>305</td>
      <td>23</td>
      <td>845</td>
      <td>574</td>
      <td>930</td>
      <td>149</td>
      <td>220</td>
      <td>208</td>
      <td>2278</td>
    </tr>
    <tr>
      <th>221957</th>
      <td>2960</td>
      <td>295</td>
      <td>21</td>
      <td>872</td>
      <td>587</td>
      <td>959</td>
      <td>155</td>
      <td>229</td>
      <td>212</td>
      <td>2285</td>
    </tr>
    <tr>
      <th>222355</th>
      <td>2948</td>
      <td>311</td>
      <td>23</td>
      <td>832</td>
      <td>576</td>
      <td>914</td>
      <td>151</td>
      <td>217</td>
      <td>202</td>
      <td>2242</td>
    </tr>
    <tr>
      <th>222356</th>
      <td>2956</td>
      <td>308</td>
      <td>19</td>
      <td>859</td>
      <td>585</td>
      <td>942</td>
      <td>165</td>
      <td>225</td>
      <td>199</td>
      <td>2249</td>
    </tr>
    <tr>
      <th>222357</th>
      <td>2964</td>
      <td>296</td>
      <td>17</td>
      <td>886</td>
      <td>589</td>
      <td>969</td>
      <td>170</td>
      <td>233</td>
      <td>202</td>
      <td>2256</td>
    </tr>
    <tr>
      <th>222774</th>
      <td>2946</td>
      <td>302</td>
      <td>26</td>
      <td>819</td>
      <td>574</td>
      <td>899</td>
      <td>138</td>
      <td>217</td>
      <td>215</td>
      <td>2206</td>
    </tr>
    <tr>
      <th>222775</th>
      <td>2956</td>
      <td>310</td>
      <td>20</td>
      <td>845</td>
      <td>586</td>
      <td>926</td>
      <td>163</td>
      <td>223</td>
      <td>198</td>
      <td>2213</td>
    </tr>
    <tr>
      <th>222776</th>
      <td>2962</td>
      <td>311</td>
      <td>15</td>
      <td>872</td>
      <td>590</td>
      <td>953</td>
      <td>179</td>
      <td>228</td>
      <td>189</td>
      <td>2219</td>
    </tr>
    <tr>
      <th>222777</th>
      <td>2968</td>
      <td>297</td>
      <td>12</td>
      <td>899</td>
      <td>597</td>
      <td>981</td>
      <td>186</td>
      <td>236</td>
      <td>190</td>
      <td>2226</td>
    </tr>
    <tr>
      <th>223207</th>
      <td>2953</td>
      <td>288</td>
      <td>23</td>
      <td>834</td>
      <td>578</td>
      <td>912</td>
      <td>150</td>
      <td>231</td>
      <td>218</td>
      <td>2177</td>
    </tr>
    <tr>
      <th>223208</th>
      <td>2962</td>
      <td>295</td>
      <td>15</td>
      <td>860</td>
      <td>590</td>
      <td>939</td>
      <td>178</td>
      <td>235</td>
      <td>197</td>
      <td>2183</td>
    </tr>
    <tr>
      <th>223209</th>
      <td>2967</td>
      <td>299</td>
      <td>10</td>
      <td>886</td>
      <td>597</td>
      <td>966</td>
      <td>193</td>
      <td>237</td>
      <td>184</td>
      <td>2190</td>
    </tr>
    <tr>
      <th>223210</th>
      <td>2970</td>
      <td>291</td>
      <td>7</td>
      <td>912</td>
      <td>598</td>
      <td>993</td>
      <td>199</td>
      <td>239</td>
      <td>179</td>
      <td>2197</td>
    </tr>
    <tr>
      <th>223652</th>
      <td>2954</td>
      <td>269</td>
      <td>21</td>
      <td>849</td>
      <td>577</td>
      <td>927</td>
      <td>160</td>
      <td>242</td>
      <td>219</td>
      <td>2148</td>
    </tr>
    <tr>
      <th>223653</th>
      <td>2963</td>
      <td>269</td>
      <td>13</td>
      <td>875</td>
      <td>588</td>
      <td>953</td>
      <td>186</td>
      <td>244</td>
      <td>197</td>
      <td>2154</td>
    </tr>
    <tr>
      <th>223654</th>
      <td>2967</td>
      <td>272</td>
      <td>8</td>
      <td>900</td>
      <td>595</td>
      <td>979</td>
      <td>199</td>
      <td>243</td>
      <td>182</td>
      <td>2161</td>
    </tr>
    <tr>
      <th>223655</th>
      <td>2971</td>
      <td>275</td>
      <td>6</td>
      <td>927</td>
      <td>601</td>
      <td>1006</td>
      <td>205</td>
      <td>241</td>
      <td>176</td>
      <td>2168</td>
    </tr>
    <tr>
      <th>223885</th>
      <td>2506</td>
      <td>13</td>
      <td>64</td>
      <td>201</td>
      <td>88</td>
      <td>655</td>
      <td>73</td>
      <td>30</td>
      <td>0</td>
      <td>1470</td>
    </tr>
    <tr>
      <th>223886</th>
      <td>2501</td>
      <td>3</td>
      <td>63</td>
      <td>216</td>
      <td>81</td>
      <td>626</td>
      <td>55</td>
      <td>40</td>
      <td>0</td>
      <td>1470</td>
    </tr>
    <tr>
      <th>223887</th>
      <td>2500</td>
      <td>0</td>
      <td>62</td>
      <td>234</td>
      <td>83</td>
      <td>598</td>
      <td>54</td>
      <td>45</td>
      <td>67</td>
      <td>1471</td>
    </tr>
    <tr>
      <th>224109</th>
      <td>2952</td>
      <td>259</td>
      <td>22</td>
      <td>865</td>
      <td>573</td>
      <td>942</td>
      <td>161</td>
      <td>246</td>
      <td>220</td>
      <td>2118</td>
    </tr>
    <tr>
      <th>224110</th>
      <td>2962</td>
      <td>261</td>
      <td>15</td>
      <td>890</td>
      <td>585</td>
      <td>967</td>
      <td>183</td>
      <td>247</td>
      <td>202</td>
      <td>2125</td>
    </tr>
    <tr>
      <th>224111</th>
      <td>2967</td>
      <td>270</td>
      <td>9</td>
      <td>916</td>
      <td>592</td>
      <td>994</td>
      <td>197</td>
      <td>243</td>
      <td>186</td>
      <td>2132</td>
    </tr>
    <tr>
      <th>224112</th>
      <td>2971</td>
      <td>285</td>
      <td>7</td>
      <td>942</td>
      <td>599</td>
      <td>1020</td>
      <td>202</td>
      <td>240</td>
      <td>178</td>
      <td>2139</td>
    </tr>
    <tr>
      <th>224573</th>
      <td>2949</td>
      <td>259</td>
      <td>24</td>
      <td>882</td>
      <td>574</td>
      <td>957</td>
      <td>156</td>
      <td>245</td>
      <td>223</td>
      <td>2089</td>
    </tr>
    <tr>
      <th>224574</th>
      <td>2960</td>
      <td>262</td>
      <td>17</td>
      <td>907</td>
      <td>581</td>
      <td>983</td>
      <td>175</td>
      <td>246</td>
      <td>209</td>
      <td>2096</td>
    </tr>
    <tr>
      <th>224575</th>
      <td>2968</td>
      <td>274</td>
      <td>12</td>
      <td>932</td>
      <td>591</td>
      <td>1008</td>
      <td>189</td>
      <td>243</td>
      <td>193</td>
      <td>2103</td>
    </tr>
    <tr>
      <th>224576</th>
      <td>2972</td>
      <td>294</td>
      <td>9</td>
      <td>957</td>
      <td>597</td>
      <td>1034</td>
      <td>194</td>
      <td>238</td>
      <td>184</td>
      <td>2110</td>
    </tr>
    <tr>
      <th>225045</th>
      <td>2959</td>
      <td>269</td>
      <td>19</td>
      <td>924</td>
      <td>584</td>
      <td>999</td>
      <td>167</td>
      <td>243</td>
      <td>214</td>
      <td>2067</td>
    </tr>
    <tr>
      <th>225046</th>
      <td>2968</td>
      <td>277</td>
      <td>15</td>
      <td>949</td>
      <td>589</td>
      <td>1024</td>
      <td>179</td>
      <td>242</td>
      <td>202</td>
      <td>2074</td>
    </tr>
    <tr>
      <th>225047</th>
      <td>2975</td>
      <td>289</td>
      <td>11</td>
      <td>973</td>
      <td>598</td>
      <td>1050</td>
      <td>190</td>
      <td>239</td>
      <td>189</td>
      <td>2081</td>
    </tr>
    <tr>
      <th>225517</th>
      <td>2959</td>
      <td>275</td>
      <td>21</td>
      <td>942</td>
      <td>582</td>
      <td>1015</td>
      <td>159</td>
      <td>240</td>
      <td>218</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>225518</th>
      <td>2970</td>
      <td>281</td>
      <td>17</td>
      <td>960</td>
      <td>595</td>
      <td>1040</td>
      <td>172</td>
      <td>239</td>
      <td>206</td>
      <td>2045</td>
    </tr>
    <tr>
      <th>479525</th>
      <td>3159</td>
      <td>60</td>
      <td>37</td>
      <td>150</td>
      <td>0</td>
      <td>3045</td>
      <td>220</td>
      <td>0</td>
      <td>17</td>
      <td>1177</td>
    </tr>
    <tr>
      <th>479789</th>
      <td>3281</td>
      <td>38</td>
      <td>59</td>
      <td>150</td>
      <td>123</td>
      <td>3012</td>
      <td>137</td>
      <td>42</td>
      <td>0</td>
      <td>1159</td>
    </tr>
    <tr>
      <th>479790</th>
      <td>3158</td>
      <td>73</td>
      <td>62</td>
      <td>170</td>
      <td>-4</td>
      <td>3042</td>
      <td>191</td>
      <td>0</td>
      <td>0</td>
      <td>1187</td>
    </tr>
    <tr>
      <th>480340</th>
      <td>3147</td>
      <td>96</td>
      <td>59</td>
      <td>216</td>
      <td>-6</td>
      <td>3037</td>
      <td>220</td>
      <td>0</td>
      <td>0</td>
      <td>1209</td>
    </tr>
    <tr>
      <th>482917</th>
      <td>3094</td>
      <td>82</td>
      <td>65</td>
      <td>42</td>
      <td>3</td>
      <td>3001</td>
      <td>193</td>
      <td>0</td>
      <td>0</td>
      <td>1315</td>
    </tr>
    <tr>
      <th>483577</th>
      <td>3083</td>
      <td>105</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>3002</td>
      <td>228</td>
      <td>0</td>
      <td>0</td>
      <td>1350</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fields that their value is >= 9
outliers_fields = np.abs(outliers-numeric_df.mean())/numeric_df.std()
outliers_fields
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>hDistance_to_Hydrology</th>
      <th>vDistance_to_Hydrology</th>
      <th>hDistance_to_Roads</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>hDistance_to_Fire Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>220084</th>
      <td>0.019163</td>
      <td>1.200418</td>
      <td>2.256377</td>
      <td>2.707944</td>
      <td>9.170238</td>
      <td>0.905013</td>
      <td>3.404797</td>
      <td>0.218462</td>
      <td>2.285377</td>
      <td>0.345651</td>
    </tr>
    <tr>
      <th>220445</th>
      <td>0.002267</td>
      <td>1.164676</td>
      <td>1.722206</td>
      <td>2.750287</td>
      <td>9.290316</td>
      <td>0.896035</td>
      <td>2.919177</td>
      <td>0.135633</td>
      <td>2.206996</td>
      <td>0.322995</td>
    </tr>
    <tr>
      <th>220812</th>
      <td>0.012982</td>
      <td>1.102128</td>
      <td>1.188035</td>
      <td>2.797335</td>
      <td>9.307470</td>
      <td>0.891545</td>
      <td>2.246780</td>
      <td>0.641483</td>
      <td>2.050234</td>
      <td>0.301095</td>
    </tr>
    <tr>
      <th>221187</th>
      <td>0.037021</td>
      <td>1.182547</td>
      <td>1.855749</td>
      <td>2.717354</td>
      <td>9.033005</td>
      <td>0.904372</td>
      <td>2.993888</td>
      <td>0.085048</td>
      <td>2.206996</td>
      <td>0.273909</td>
    </tr>
    <tr>
      <th>221188</th>
      <td>0.012982</td>
      <td>1.137869</td>
      <td>0.653865</td>
      <td>2.844383</td>
      <td>9.290316</td>
      <td>0.886415</td>
      <td>1.798515</td>
      <td>0.692068</td>
      <td>1.815091</td>
      <td>0.279195</td>
    </tr>
    <tr>
      <th>221567</th>
      <td>0.015591</td>
      <td>1.227224</td>
      <td>1.455121</td>
      <td>2.773811</td>
      <td>9.204546</td>
      <td>0.898600</td>
      <td>2.582979</td>
      <td>0.085048</td>
      <td>1.997980</td>
      <td>0.252009</td>
    </tr>
    <tr>
      <th>221956</th>
      <td>0.037021</td>
      <td>1.334449</td>
      <td>1.188035</td>
      <td>2.707944</td>
      <td>9.050160</td>
      <td>0.910785</td>
      <td>2.358846</td>
      <td>0.167877</td>
      <td>1.710582</td>
      <td>0.224822</td>
    </tr>
    <tr>
      <th>221957</th>
      <td>0.002267</td>
      <td>1.245095</td>
      <td>0.920950</td>
      <td>2.834973</td>
      <td>9.273162</td>
      <td>0.892187</td>
      <td>2.134714</td>
      <td>0.287388</td>
      <td>1.815091</td>
      <td>0.230109</td>
    </tr>
    <tr>
      <th>222355</th>
      <td>0.040593</td>
      <td>1.388062</td>
      <td>1.188035</td>
      <td>2.646782</td>
      <td>9.084468</td>
      <td>0.921047</td>
      <td>2.284135</td>
      <td>0.319632</td>
      <td>1.553820</td>
      <td>0.197636</td>
    </tr>
    <tr>
      <th>222356</th>
      <td>0.012020</td>
      <td>1.361256</td>
      <td>0.653865</td>
      <td>2.773811</td>
      <td>9.238854</td>
      <td>0.903089</td>
      <td>1.761160</td>
      <td>0.085048</td>
      <td>1.475439</td>
      <td>0.202922</td>
    </tr>
    <tr>
      <th>222357</th>
      <td>0.016553</td>
      <td>1.254030</td>
      <td>0.386779</td>
      <td>2.900841</td>
      <td>9.307470</td>
      <td>0.885773</td>
      <td>1.574383</td>
      <td>0.489728</td>
      <td>1.553820</td>
      <td>0.208209</td>
    </tr>
    <tr>
      <th>222774</th>
      <td>0.047736</td>
      <td>1.307643</td>
      <td>1.588664</td>
      <td>2.585620</td>
      <td>9.050160</td>
      <td>0.930667</td>
      <td>2.769756</td>
      <td>0.319632</td>
      <td>1.893472</td>
      <td>0.170450</td>
    </tr>
    <tr>
      <th>222775</th>
      <td>0.012020</td>
      <td>1.379127</td>
      <td>0.787407</td>
      <td>2.707944</td>
      <td>9.256008</td>
      <td>0.913351</td>
      <td>1.835870</td>
      <td>0.016122</td>
      <td>1.449312</td>
      <td>0.175736</td>
    </tr>
    <tr>
      <th>222776</th>
      <td>0.009410</td>
      <td>1.388062</td>
      <td>0.119694</td>
      <td>2.834973</td>
      <td>9.324624</td>
      <td>0.896035</td>
      <td>1.238184</td>
      <td>0.236803</td>
      <td>1.214169</td>
      <td>0.180267</td>
    </tr>
    <tr>
      <th>222777</th>
      <td>0.030840</td>
      <td>1.262966</td>
      <td>0.280934</td>
      <td>2.962003</td>
      <td>9.444703</td>
      <td>0.878077</td>
      <td>0.976696</td>
      <td>0.641483</td>
      <td>1.240296</td>
      <td>0.185553</td>
    </tr>
    <tr>
      <th>223207</th>
      <td>0.022734</td>
      <td>1.182547</td>
      <td>1.188035</td>
      <td>2.656191</td>
      <td>9.118776</td>
      <td>0.922329</td>
      <td>2.321491</td>
      <td>0.388558</td>
      <td>1.971853</td>
      <td>0.148550</td>
    </tr>
    <tr>
      <th>223208</th>
      <td>0.009410</td>
      <td>1.245095</td>
      <td>0.119694</td>
      <td>2.778516</td>
      <td>9.324624</td>
      <td>0.905013</td>
      <td>1.275539</td>
      <td>0.590898</td>
      <td>1.423185</td>
      <td>0.153081</td>
    </tr>
    <tr>
      <th>223209</th>
      <td>0.027268</td>
      <td>1.280837</td>
      <td>0.548020</td>
      <td>2.900841</td>
      <td>9.444703</td>
      <td>0.887697</td>
      <td>0.715208</td>
      <td>0.692068</td>
      <td>1.083534</td>
      <td>0.158367</td>
    </tr>
    <tr>
      <th>223210</th>
      <td>0.037983</td>
      <td>1.209353</td>
      <td>0.948648</td>
      <td>3.023165</td>
      <td>9.461857</td>
      <td>0.870382</td>
      <td>0.491076</td>
      <td>0.793238</td>
      <td>0.952898</td>
      <td>0.163653</td>
    </tr>
    <tr>
      <th>223652</th>
      <td>0.019163</td>
      <td>1.012773</td>
      <td>0.920950</td>
      <td>2.726763</td>
      <td>9.101622</td>
      <td>0.912709</td>
      <td>1.947937</td>
      <td>0.944993</td>
      <td>1.997980</td>
      <td>0.126650</td>
    </tr>
    <tr>
      <th>223653</th>
      <td>0.012982</td>
      <td>1.012773</td>
      <td>0.147392</td>
      <td>2.849088</td>
      <td>9.290316</td>
      <td>0.896035</td>
      <td>0.976696</td>
      <td>1.046163</td>
      <td>1.423185</td>
      <td>0.131181</td>
    </tr>
    <tr>
      <th>223654</th>
      <td>0.027268</td>
      <td>1.039579</td>
      <td>0.815105</td>
      <td>2.966708</td>
      <td>9.410395</td>
      <td>0.879360</td>
      <td>0.491076</td>
      <td>0.995578</td>
      <td>1.031279</td>
      <td>0.136467</td>
    </tr>
    <tr>
      <th>223655</th>
      <td>0.041555</td>
      <td>1.066386</td>
      <td>1.082190</td>
      <td>3.093737</td>
      <td>9.513319</td>
      <td>0.862044</td>
      <td>0.266944</td>
      <td>0.894408</td>
      <td>0.874517</td>
      <td>0.141753</td>
    </tr>
    <tr>
      <th>223885</th>
      <td>1.619250</td>
      <td>1.274703</td>
      <td>6.663286</td>
      <td>0.321940</td>
      <td>0.713286</td>
      <td>1.087152</td>
      <td>5.197857</td>
      <td>9.779032</td>
      <td>3.723841</td>
      <td>0.385360</td>
    </tr>
    <tr>
      <th>223886</th>
      <td>1.637108</td>
      <td>1.364058</td>
      <td>6.529743</td>
      <td>0.251369</td>
      <td>0.593207</td>
      <td>1.105750</td>
      <td>5.870254</td>
      <td>9.273181</td>
      <td>3.723841</td>
      <td>0.385360</td>
    </tr>
    <tr>
      <th>223887</th>
      <td>1.640680</td>
      <td>1.390864</td>
      <td>6.396201</td>
      <td>0.166682</td>
      <td>0.627515</td>
      <td>1.123708</td>
      <td>5.907609</td>
      <td>9.020256</td>
      <td>1.973330</td>
      <td>0.384604</td>
    </tr>
    <tr>
      <th>224109</th>
      <td>0.026306</td>
      <td>0.923418</td>
      <td>1.054493</td>
      <td>2.802040</td>
      <td>9.033005</td>
      <td>0.903089</td>
      <td>1.910581</td>
      <td>1.147333</td>
      <td>2.024107</td>
      <td>0.103994</td>
    </tr>
    <tr>
      <th>224110</th>
      <td>0.009410</td>
      <td>0.941289</td>
      <td>0.119694</td>
      <td>2.919660</td>
      <td>9.238854</td>
      <td>0.887056</td>
      <td>1.088762</td>
      <td>1.197918</td>
      <td>1.553820</td>
      <td>0.109281</td>
    </tr>
    <tr>
      <th>224111</th>
      <td>0.027268</td>
      <td>1.021708</td>
      <td>0.681562</td>
      <td>3.041984</td>
      <td>9.358933</td>
      <td>0.869740</td>
      <td>0.565787</td>
      <td>0.995578</td>
      <td>1.135788</td>
      <td>0.114567</td>
    </tr>
    <tr>
      <th>224112</th>
      <td>0.041555</td>
      <td>1.155740</td>
      <td>0.948648</td>
      <td>3.164309</td>
      <td>9.479011</td>
      <td>0.853066</td>
      <td>0.379010</td>
      <td>0.843823</td>
      <td>0.926771</td>
      <td>0.119853</td>
    </tr>
    <tr>
      <th>224573</th>
      <td>0.037021</td>
      <td>0.923418</td>
      <td>1.321578</td>
      <td>2.882021</td>
      <td>9.050160</td>
      <td>0.893469</td>
      <td>2.097358</td>
      <td>1.096748</td>
      <td>2.102488</td>
      <td>0.082094</td>
    </tr>
    <tr>
      <th>224574</th>
      <td>0.002267</td>
      <td>0.950225</td>
      <td>0.386779</td>
      <td>2.999641</td>
      <td>9.170238</td>
      <td>0.876795</td>
      <td>1.387606</td>
      <td>1.147333</td>
      <td>1.736709</td>
      <td>0.087380</td>
    </tr>
    <tr>
      <th>224575</th>
      <td>0.030840</td>
      <td>1.057450</td>
      <td>0.280934</td>
      <td>3.117261</td>
      <td>9.341779</td>
      <td>0.860762</td>
      <td>0.864630</td>
      <td>0.995578</td>
      <td>1.318677</td>
      <td>0.092667</td>
    </tr>
    <tr>
      <th>224576</th>
      <td>0.045126</td>
      <td>1.236159</td>
      <td>0.681562</td>
      <td>3.234881</td>
      <td>9.444703</td>
      <td>0.844087</td>
      <td>0.677853</td>
      <td>0.742653</td>
      <td>1.083534</td>
      <td>0.097953</td>
    </tr>
    <tr>
      <th>225045</th>
      <td>0.001305</td>
      <td>1.012773</td>
      <td>0.653865</td>
      <td>3.079623</td>
      <td>9.221700</td>
      <td>0.866534</td>
      <td>1.686449</td>
      <td>0.995578</td>
      <td>1.867345</td>
      <td>0.065480</td>
    </tr>
    <tr>
      <th>225046</th>
      <td>0.030840</td>
      <td>1.084257</td>
      <td>0.119694</td>
      <td>3.197242</td>
      <td>9.307470</td>
      <td>0.850500</td>
      <td>1.238184</td>
      <td>0.944993</td>
      <td>1.553820</td>
      <td>0.070767</td>
    </tr>
    <tr>
      <th>225047</th>
      <td>0.055841</td>
      <td>1.191482</td>
      <td>0.414477</td>
      <td>3.310157</td>
      <td>9.461857</td>
      <td>0.833826</td>
      <td>0.827275</td>
      <td>0.793238</td>
      <td>1.214169</td>
      <td>0.076053</td>
    </tr>
    <tr>
      <th>225517</th>
      <td>0.001305</td>
      <td>1.066386</td>
      <td>0.920950</td>
      <td>3.164309</td>
      <td>9.187392</td>
      <td>0.856272</td>
      <td>1.985292</td>
      <td>0.843823</td>
      <td>1.971853</td>
      <td>0.042825</td>
    </tr>
    <tr>
      <th>225518</th>
      <td>0.037983</td>
      <td>1.119998</td>
      <td>0.386779</td>
      <td>3.248995</td>
      <td>9.410395</td>
      <td>0.840239</td>
      <td>1.499672</td>
      <td>0.793238</td>
      <td>1.658328</td>
      <td>0.048866</td>
    </tr>
    <tr>
      <th>479525</th>
      <td>0.713020</td>
      <td>0.854737</td>
      <td>3.057633</td>
      <td>0.561885</td>
      <td>0.796272</td>
      <td>0.445632</td>
      <td>0.293388</td>
      <td>11.296582</td>
      <td>3.279681</td>
      <td>0.606626</td>
    </tr>
    <tr>
      <th>479789</th>
      <td>1.148758</td>
      <td>1.051317</td>
      <td>5.995572</td>
      <td>0.561885</td>
      <td>1.313678</td>
      <td>0.424468</td>
      <td>2.807111</td>
      <td>9.172011</td>
      <td>3.723841</td>
      <td>0.620219</td>
    </tr>
    <tr>
      <th>479790</th>
      <td>0.709448</td>
      <td>0.738576</td>
      <td>6.396201</td>
      <td>0.467789</td>
      <td>0.864888</td>
      <td>0.443708</td>
      <td>0.789919</td>
      <td>11.296582</td>
      <td>3.723841</td>
      <td>0.599074</td>
    </tr>
    <tr>
      <th>480340</th>
      <td>0.670160</td>
      <td>0.533061</td>
      <td>5.995572</td>
      <td>0.251369</td>
      <td>0.899196</td>
      <td>0.440501</td>
      <td>0.293388</td>
      <td>11.296582</td>
      <td>3.723841</td>
      <td>0.582460</td>
    </tr>
    <tr>
      <th>482917</th>
      <td>0.480864</td>
      <td>0.658157</td>
      <td>6.796829</td>
      <td>1.070002</td>
      <td>0.744810</td>
      <td>0.417413</td>
      <td>0.715208</td>
      <td>11.296582</td>
      <td>3.723841</td>
      <td>0.502412</td>
    </tr>
    <tr>
      <th>483577</th>
      <td>0.441577</td>
      <td>0.452642</td>
      <td>5.728487</td>
      <td>1.267603</td>
      <td>0.796272</td>
      <td>0.418054</td>
      <td>0.592231</td>
      <td>11.296582</td>
      <td>3.723841</td>
      <td>0.475981</td>
    </tr>
  </tbody>
</table>
</div>




```python
outliers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 45 entries, 220084 to 483577
    Data columns (total 10 columns):
    Elevation                   45 non-null int64
    Aspect                      45 non-null int64
    Slope                       45 non-null int64
    hDistance_to_Hydrology      45 non-null int64
    vDistance_to_Hydrology      45 non-null int64
    hDistance_to_Roads          45 non-null int64
    Hillshade_9am               45 non-null int64
    Hillshade_Noon              45 non-null int64
    Hillshade_3pm               45 non-null int64
    hDistance_to_Fire Points    45 non-null int64
    dtypes: int64(10)
    memory usage: 3.9 KB



```python
# Boxplots of quantitative attributes
%matplotlib inline
vars_to_plot_separate1 = [['Elevation'],
                         ['Aspect'], 
                         ['Slope'],
                         ['hDistance_to_Hydrology'], 
                         ['vDistance_to_Hydrology']]
                                                 
vars_to_plot_separate2 = [['hDistance_to_Roads','hDistance_to_Fire Points'],
                         ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]
plt.figure(figsize=(15, 6))
for index, plot_vars in enumerate(vars_to_plot_separate1):
    plt.subplot(len(vars_to_plot_separate1)/2, 
                3, 
                index+1)
    ax = df.boxplot(column=plot_vars) 
plt.show()

plt.figure(figsize=(15, 3))
for index, plot_vars in enumerate(vars_to_plot_separate2):
    plt.subplot(len(vars_to_plot_separate2)/2, 
                2, 
                index+1)
    ax = df.boxplot(column=plot_vars)
    plt.xticks(rotation=60)
plt.show()
```


![png](output_35_0.png)



![png](output_35_1.png)


Above we can see that many of the attributes display some heavy tails creating skew. Aspect is the only attribute not showing any outliers. This will be verified in the next section with the distribution plots.

[top](#toc)
<a id='Basic_Stats'></a>
## Basic Statistics and Visualizations

First we will consider individually the numeric data, and then the categorical data. There are some basic statistics in the numeric data which are interesting. 

First aspect which was the only attribute in the box plots to not show heavy taling has a standard deviation almost as big as its mean. Since this in degrees and is an angular measure from a reference, this would make the range from 0 t0 360. This means that all the range of values of aspect are about three standard deviations apart. Because this is a positioning measure from a reference point, this may not be significant, but the variance is noteworthy. Other attributes have standard deviations close to their mean and reach out of the interquartile range within a couple of standard deviations, but none are as extreme aspect.

The indexes for hillshade are interesting because, like aspect, the values are bounded; the index for hillshade is between 0 and 254. But unlike aspect, these indexes have a standard deviation that is much smaller than the mean and the interquartile range is pretty tight. The standard deviation is smallest for the measurement at noon when the sun is near directly overhead. This would produce the least amount of variation in the measurement and that is reflected in the values.

The last attribute that jumps out is the vertical distance to hydrology because it ranges into negative numbers. This means that the sometimes the mean distance is sometimes higher or lower than the water area.


```python
df[quantitative].describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elevation</th>
      <td>581012.0</td>
      <td>2959.365301</td>
      <td>279.984734</td>
      <td>1859.0</td>
      <td>2809.0</td>
      <td>2996.0</td>
      <td>3163.0</td>
      <td>3858.0</td>
    </tr>
    <tr>
      <th>Aspect</th>
      <td>581012.0</td>
      <td>155.656807</td>
      <td>111.913721</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>127.0</td>
      <td>260.0</td>
      <td>360.0</td>
    </tr>
    <tr>
      <th>Slope</th>
      <td>581012.0</td>
      <td>14.103704</td>
      <td>7.488242</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>hDistance_to_Hydrology</th>
      <td>581012.0</td>
      <td>269.428217</td>
      <td>212.549356</td>
      <td>0.0</td>
      <td>108.0</td>
      <td>218.0</td>
      <td>384.0</td>
      <td>1397.0</td>
    </tr>
    <tr>
      <th>vDistance_to_Hydrology</th>
      <td>581012.0</td>
      <td>46.418855</td>
      <td>58.295232</td>
      <td>-173.0</td>
      <td>7.0</td>
      <td>30.0</td>
      <td>69.0</td>
      <td>601.0</td>
    </tr>
    <tr>
      <th>hDistance_to_Roads</th>
      <td>581012.0</td>
      <td>2350.146611</td>
      <td>1559.254870</td>
      <td>0.0</td>
      <td>1106.0</td>
      <td>1997.0</td>
      <td>3328.0</td>
      <td>7117.0</td>
    </tr>
    <tr>
      <th>Hillshade_9am</th>
      <td>581012.0</td>
      <td>212.146049</td>
      <td>26.769889</td>
      <td>0.0</td>
      <td>198.0</td>
      <td>218.0</td>
      <td>231.0</td>
      <td>254.0</td>
    </tr>
    <tr>
      <th>Hillshade_Noon</th>
      <td>581012.0</td>
      <td>223.318716</td>
      <td>19.768697</td>
      <td>0.0</td>
      <td>213.0</td>
      <td>226.0</td>
      <td>237.0</td>
      <td>254.0</td>
    </tr>
    <tr>
      <th>Hillshade_3pm</th>
      <td>581012.0</td>
      <td>142.528263</td>
      <td>38.274529</td>
      <td>0.0</td>
      <td>119.0</td>
      <td>143.0</td>
      <td>168.0</td>
      <td>254.0</td>
    </tr>
    <tr>
      <th>hDistance_to_Fire Points</th>
      <td>581012.0</td>
      <td>1980.291226</td>
      <td>1324.195210</td>
      <td>0.0</td>
      <td>1024.0</td>
      <td>1710.0</td>
      <td>2550.0</td>
      <td>7173.0</td>
    </tr>
  </tbody>
</table>
</div>



Distribution plots of all the numeric data were performed and it verifies what we saw in the box plots and in the statistical summaries. The variation for aspect is big and there is heavy tailing present for most of the other distributions. Only the amount of light at 3:00 PM looks approximately normal. This real data emphasizes the importance of the central limit theorem if needing to do an analysis where one of the assumptions is normality.


```python
%matplotlib inline
plt.figure(figsize=(10,3))
for i, col in enumerate(df[quantitative]):
    plt.figure(i)
    sns.distplot(df[quantitative][col])
```


![png](output_41_0.png)



![png](output_41_1.png)



![png](output_41_2.png)



![png](output_41_3.png)



![png](output_41_4.png)



![png](output_41_5.png)



![png](output_41_6.png)



![png](output_41_7.png)



![png](output_41_8.png)



![png](output_41_9.png)


For the categorical data individual analysis is done with basic frequency counts displayed by bar graphs. The first graph clearly shows that most of the data comes from the Rawah and Comanch Peak wilderness areas. This is expected because the acreage from those two areas is larger and thus there can be more 30 x 30 m cells.


```python
%matplotlib inline
plt.figure(figsize=(5,3))
ax = sns.countplot(x="Wilderness_Area",data=df, palette="Blues_d")
ax.set_title('Wilderness Area Frequency')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3]), <a list of 4 Text xticklabel objects>)




![png](output_43_1.png)


The next one is soil type, and though there are forty soil types, from the graph, we can estimate the most of the records come from about 25% of the soil types. The most prevalent soil type is number 29 which correlates to: Como - Legault families complex, extremely stony Since this is a stony area, it would be interesting in the relations section to see what wilderness area and tree type grow from this soil. The graph is slightly offset making it look like soil type 28, but running the numbers with a cross-tab show that it should be soil type 29.


```python
%matplotlib inline
plt.figure(figsize=(10,3))
ax = sns.countplot(x="Soil_Type",data=df, palette="Blues_d")
ax.set_title('Soil Type Frequency')
plt.xticks(rotation=60)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39]), <a list of 40 Text xticklabel objects>)




![png](output_45_1.png)


The bar graph shows that a majority of the records come from the subalpine climate zone. The subalpine zone is just below the tree line at high elevations (9000-12000 ft.) and is cool year round.


```python
%matplotlib inline
plt.figure(figsize=(8,3))
ax = sns.countplot(x="Climatic_Zone",data=df, palette="Blues_d")
ax.set_title('Climatic Zone Frequency')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3, 4, 5, 6]), <a list of 7 Text xticklabel objects>)




![png](output_47_1.png)


There is mostly igneous and metamorphic rock in from these wilderness areas. This corresponds with the prevalence of soil type 29 seen in the soil type chart.


```python
%matplotlib inline
plt.figure(figsize=(5,3))
ax = sns.countplot(x="Geologic_Zone",data=df, palette="Blues_d")
ax.set_title('Geologic Zone Frequency')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3]), <a list of 4 Text xticklabel objects>)




![png](output_49_1.png)


The graph shows the most of the trees in these wilderness areas are Lodgepole Pine, and Spruce/Fir. This probably means that the Rawah and Comanche Peak areas predominantly have these types. It would be interesting to see whether the other tree types could be predicted with better or worse accuracy than these two.


```python
%matplotlib inline
plt.figure(figsize=(9,3))
ax = sns.countplot(x="Cover_Type",data=df, palette="Blues_d")
ax.set_title('Cover Type Frequency')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3, 4, 5, 6]), <a list of 7 Text xticklabel objects>)




![png](output_51_1.png)


[top](#toc)
<a id='Attribute_Interest'></a>
## Attributes of Interest

#### Cover Type

The cover type is what the other attributes are trying to predict and would be the response for the analysis, but it would be interesting to look at an overall numerical breakdown of tree types. We already saw that Lodgepine and Spruce/Fir occured more frequently, but a percentage breakdown would complete the picture. It would also be interesting to see the most frequent tree types in each wilderness area, climatic zone, and at what elevation.


```python
pd.crosstab(df.Cover_Type, df.Wilderness_Area, margins=True, margins_name="Total")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aspen</th>
      <td>0</td>
      <td>5712</td>
      <td>0</td>
      <td>3781</td>
      <td>9493</td>
    </tr>
    <tr>
      <th>Cottonwood/Willow</th>
      <td>2747</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2747</td>
    </tr>
    <tr>
      <th>Douglas-fir</th>
      <td>9741</td>
      <td>7626</td>
      <td>0</td>
      <td>0</td>
      <td>17367</td>
    </tr>
    <tr>
      <th>Krummholz</th>
      <td>0</td>
      <td>13105</td>
      <td>2304</td>
      <td>5101</td>
      <td>20510</td>
    </tr>
    <tr>
      <th>Lodgepole Pine</th>
      <td>3026</td>
      <td>125093</td>
      <td>8985</td>
      <td>146197</td>
      <td>283301</td>
    </tr>
    <tr>
      <th>Ponderosa Pine</th>
      <td>21454</td>
      <td>14300</td>
      <td>0</td>
      <td>0</td>
      <td>35754</td>
    </tr>
    <tr>
      <th>Spruce/Fir</th>
      <td>0</td>
      <td>87528</td>
      <td>18595</td>
      <td>105717</td>
      <td>211840</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>36968</td>
      <td>253364</td>
      <td>29884</td>
      <td>260796</td>
      <td>581012</td>
    </tr>
  </tbody>
</table>
</div>



This a breakdown of the trees in the various wilderness areas by count. Noticeable immediately are the zeros. The cottonwood/willow cover type only occurs in Cache la Poudre. The douglas fir does not occur in Neota and Rawah. This is curious because the biggest areas are Rawah and Comanche Peak, so there must be some other condition difference between Rawah and Comanch Peak that lets the tree only grow in one and not the other. The same circumstance is seen with the Ponderosa Pine. The spruce fir which is the second most populous tree on the list does not occur in Cache la Poudre. Cache la Poudre is looking a little exclusive regarding cover types because neither Krummholz nor Aspen grow there. The numbers here also confirm what the graph showed which is that there are much more Lodgepole Pines and Spruce/Fir cover types.


```python
pd.crosstab(df.Cover_Type, df.Wilderness_Area, margins=True, margins_name="Total", normalize="index")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aspen</th>
      <td>0.000000</td>
      <td>0.601707</td>
      <td>0.000000</td>
      <td>0.398293</td>
    </tr>
    <tr>
      <th>Cottonwood/Willow</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Douglas-fir</th>
      <td>0.560891</td>
      <td>0.439109</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Krummholz</th>
      <td>0.000000</td>
      <td>0.638957</td>
      <td>0.112335</td>
      <td>0.248708</td>
    </tr>
    <tr>
      <th>Lodgepole Pine</th>
      <td>0.010681</td>
      <td>0.441555</td>
      <td>0.031715</td>
      <td>0.516048</td>
    </tr>
    <tr>
      <th>Ponderosa Pine</th>
      <td>0.600045</td>
      <td>0.399955</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Spruce/Fir</th>
      <td>0.000000</td>
      <td>0.413180</td>
      <td>0.087779</td>
      <td>0.499042</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.063627</td>
      <td>0.436074</td>
      <td>0.051434</td>
      <td>0.448865</td>
    </tr>
  </tbody>
</table>
</div>



Normalizing the tree type numbers gives the percentage breakdown of tree type per wilderness area by the total of each individual tree type. All the rows should add up to 100%. The majority of the total number of trees are in the Rawah and Comanche Peak area, but these are the biggest areas. The Douglas Fir and Ponderosa Pine are about evenly split between Cache la Poudre and Comanche Peak. The biggest occurrence is with Krummholz where 64% of its trees are in Comanche Peak. The smallest occurrence is with the Lodgepole Pine. Only 1% of it's trees are found in Cache la Poudre.


```python
pd.crosstab(df.Cover_Type, df.Wilderness_Area, margins=True, margins_name="Total", normalize="columns")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aspen</th>
      <td>0.000000</td>
      <td>0.022545</td>
      <td>0.000000</td>
      <td>0.014498</td>
      <td>0.016339</td>
    </tr>
    <tr>
      <th>Cottonwood/Willow</th>
      <td>0.074308</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004728</td>
    </tr>
    <tr>
      <th>Douglas-fir</th>
      <td>0.263498</td>
      <td>0.030099</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029891</td>
    </tr>
    <tr>
      <th>Krummholz</th>
      <td>0.000000</td>
      <td>0.051724</td>
      <td>0.077098</td>
      <td>0.019559</td>
      <td>0.035300</td>
    </tr>
    <tr>
      <th>Lodgepole Pine</th>
      <td>0.081855</td>
      <td>0.493728</td>
      <td>0.300663</td>
      <td>0.560580</td>
      <td>0.487599</td>
    </tr>
    <tr>
      <th>Ponderosa Pine</th>
      <td>0.580340</td>
      <td>0.056441</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061537</td>
    </tr>
    <tr>
      <th>Spruce/Fir</th>
      <td>0.000000</td>
      <td>0.345463</td>
      <td>0.622239</td>
      <td>0.405363</td>
      <td>0.364605</td>
    </tr>
  </tbody>
</table>
</div>



The next table is similar except it normalizes the tree types according to the wilderness area numbers so that we can see the breakdown of trees within each area. Each column should add to 100%. Between this table and the table normalized according to row, a better distribution of the trees throughout the wilderness area can be seen. The other important item from this table is that the total column gives the fraction from all the tree types. Therefore, the Lodgepol Pine makes 49% of all the tree types recorded. The Cottnwood/Willow is less than one percent.

#### Soil Type

This categorical variable has forty values, and since the type of soil can affect the plant life, it would be good to get a handle on some of the numbers.


```python
pd.crosstab(df.Soil_Type, df.Wilderness_Area, margins=True, margins_name="Total")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Soil_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Soil Type 1</th>
      <td>3031</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>Soil Type 10</th>
      <td>17914</td>
      <td>14720</td>
      <td>0</td>
      <td>0</td>
      <td>32634</td>
    </tr>
    <tr>
      <th>Soil Type 11</th>
      <td>596</td>
      <td>11814</td>
      <td>0</td>
      <td>0</td>
      <td>12410</td>
    </tr>
    <tr>
      <th>Soil Type 12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29971</td>
      <td>29971</td>
    </tr>
    <tr>
      <th>Soil Type 13</th>
      <td>0</td>
      <td>17176</td>
      <td>255</td>
      <td>0</td>
      <td>17431</td>
    </tr>
    <tr>
      <th>Soil Type 14</th>
      <td>359</td>
      <td>240</td>
      <td>0</td>
      <td>0</td>
      <td>599</td>
    </tr>
    <tr>
      <th>Soil Type 15</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Soil Type 16</th>
      <td>263</td>
      <td>325</td>
      <td>117</td>
      <td>2140</td>
      <td>2845</td>
    </tr>
    <tr>
      <th>Soil Type 17</th>
      <td>793</td>
      <td>2629</td>
      <td>0</td>
      <td>0</td>
      <td>3422</td>
    </tr>
    <tr>
      <th>Soil Type 18</th>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>1829</td>
      <td>1899</td>
    </tr>
    <tr>
      <th>Soil Type 19</th>
      <td>0</td>
      <td>675</td>
      <td>597</td>
      <td>2749</td>
      <td>4021</td>
    </tr>
    <tr>
      <th>Soil Type 2</th>
      <td>2144</td>
      <td>5381</td>
      <td>0</td>
      <td>0</td>
      <td>7525</td>
    </tr>
    <tr>
      <th>Soil Type 20</th>
      <td>0</td>
      <td>2452</td>
      <td>55</td>
      <td>6752</td>
      <td>9259</td>
    </tr>
    <tr>
      <th>Soil Type 21</th>
      <td>0</td>
      <td>838</td>
      <td>0</td>
      <td>0</td>
      <td>838</td>
    </tr>
    <tr>
      <th>Soil Type 22</th>
      <td>0</td>
      <td>8362</td>
      <td>5363</td>
      <td>19648</td>
      <td>33373</td>
    </tr>
    <tr>
      <th>Soil Type 23</th>
      <td>0</td>
      <td>21071</td>
      <td>8153</td>
      <td>28528</td>
      <td>57752</td>
    </tr>
    <tr>
      <th>Soil Type 24</th>
      <td>0</td>
      <td>16252</td>
      <td>2123</td>
      <td>2903</td>
      <td>21278</td>
    </tr>
    <tr>
      <th>Soil Type 25</th>
      <td>0</td>
      <td>0</td>
      <td>474</td>
      <td>0</td>
      <td>474</td>
    </tr>
    <tr>
      <th>Soil Type 26</th>
      <td>0</td>
      <td>2589</td>
      <td>0</td>
      <td>0</td>
      <td>2589</td>
    </tr>
    <tr>
      <th>Soil Type 27</th>
      <td>0</td>
      <td>1086</td>
      <td>0</td>
      <td>0</td>
      <td>1086</td>
    </tr>
    <tr>
      <th>Soil Type 28</th>
      <td>0</td>
      <td>946</td>
      <td>0</td>
      <td>0</td>
      <td>946</td>
    </tr>
    <tr>
      <th>Soil Type 29</th>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>115173</td>
      <td>115247</td>
    </tr>
    <tr>
      <th>Soil Type 3</th>
      <td>2455</td>
      <td>2368</td>
      <td>0</td>
      <td>0</td>
      <td>4823</td>
    </tr>
    <tr>
      <th>Soil Type 30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30170</td>
      <td>30170</td>
    </tr>
    <tr>
      <th>Soil Type 31</th>
      <td>0</td>
      <td>25240</td>
      <td>426</td>
      <td>0</td>
      <td>25666</td>
    </tr>
    <tr>
      <th>Soil Type 32</th>
      <td>0</td>
      <td>48758</td>
      <td>3761</td>
      <td>0</td>
      <td>52519</td>
    </tr>
    <tr>
      <th>Soil Type 33</th>
      <td>0</td>
      <td>42337</td>
      <td>2817</td>
      <td>0</td>
      <td>45154</td>
    </tr>
    <tr>
      <th>Soil Type 34</th>
      <td>0</td>
      <td>1611</td>
      <td>0</td>
      <td>0</td>
      <td>1611</td>
    </tr>
    <tr>
      <th>Soil Type 35</th>
      <td>0</td>
      <td>732</td>
      <td>503</td>
      <td>656</td>
      <td>1891</td>
    </tr>
    <tr>
      <th>Soil Type 36</th>
      <td>0</td>
      <td>119</td>
      <td>0</td>
      <td>0</td>
      <td>119</td>
    </tr>
    <tr>
      <th>Soil Type 37</th>
      <td>0</td>
      <td>66</td>
      <td>0</td>
      <td>232</td>
      <td>298</td>
    </tr>
    <tr>
      <th>Soil Type 38</th>
      <td>0</td>
      <td>5993</td>
      <td>2073</td>
      <td>7507</td>
      <td>15573</td>
    </tr>
    <tr>
      <th>Soil Type 39</th>
      <td>0</td>
      <td>6117</td>
      <td>931</td>
      <td>6758</td>
      <td>13806</td>
    </tr>
    <tr>
      <th>Soil Type 4</th>
      <td>1238</td>
      <td>11158</td>
      <td>0</td>
      <td>0</td>
      <td>12396</td>
    </tr>
    <tr>
      <th>Soil Type 40</th>
      <td>0</td>
      <td>2309</td>
      <td>2092</td>
      <td>4349</td>
      <td>8750</td>
    </tr>
    <tr>
      <th>Soil Type 5</th>
      <td>1597</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1597</td>
    </tr>
    <tr>
      <th>Soil Type 6</th>
      <td>6575</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6575</td>
    </tr>
    <tr>
      <th>Soil Type 7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>Soil Type 8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>179</td>
      <td>179</td>
    </tr>
    <tr>
      <th>Soil Type 9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1147</td>
      <td>1147</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>36968</td>
      <td>253364</td>
      <td>29884</td>
      <td>260796</td>
      <td>581012</td>
    </tr>
  </tbody>
</table>
</div>



So the big numbers and little numbers are easily determined. Soil type 29 dominates the numbers and soil type 15 only has three occurrences. Soil type 15 is unspecified. The next lowest number is soil type 7 which is the gothic family.
Because there are so many values, it makes it a little harder to read so we will make a partial visual plot to help.


```python
plt.figure(figsize=(10,20))
sns.heatmap(pd.crosstab(df.Soil_Type, df.Wilderness_Area, normalize="columns", margins=True, margins_name="Total"),
            cmap="YlGnBu", annot=True, cbar=False)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb5dc12c7b8>




![png](output_63_1.png)


The darker colors highlight higher numbers with the level of darkness proportional to the magnitude of percentage. This map makes it a little easier to see that soil type 29 is 20% of all the soil types throughout the wilderness areas, but it is 44% of the soil types in the Rawah area. Soil type 10 is 48% of all the soil in Cache la Poudre. We know from the previous cover type analysis is that this area is more exclusive and is largely made up of Ponderosa Pine. Soil type 10 however is only 6% of the total soil types in all the areas. This makes sense that exclusivity of Cache la Poudre in soil type agrees with its exclusivity in tree type.

#### Aspect

Aspect is the azimuth measured in degrees from a reference point, so it is horizontal angular distance to some reference. This is a positional attribute, but what makes it interesting is its variance. The total range of degrees from 0 to 360 is covered in 3 standard deviations. We'll look at violin plots to visualize this.


```python
sns.violinplot(data = df, x="Wilderness_Area", y="Aspect")
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3]), <a list of 4 Text xticklabel objects>)




![png](output_66_1.png)


It looks like the distribution is largely bidmodal for all the areas, but if this is an angular distance from a reference line, then interestingly enough, this should represent a positional clustering of areas. This suggests not as many cells across the wilderness areas at 200 degrees from reference, and most around 0 and 360 degrees which are right at the reference line. This probably means that the reference line for this measure is drawn right through the area with the most cells.

#### Elevation

The previous analysis have shown some exclusivity with cover types, soil types, and wilderness area, so it would be interesting to look at the elevation of these areas.



```python
plt.figure(figsize=(10,5))
ax = sns.boxplot(x='Wilderness_Area',y='Elevation',data=df, palette = "Blues_d")
ax.set_title('Elevation by Cover Type')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3]), <a list of 4 Text xticklabel objects>)




![png](output_69_1.png)


The boxplot shows that the area showing the most exclusivity is also the one with the lowest mean elevation. The two largest areas which have the most number of cover types are at about the same elevation. The neota area has the highest elevation

#### Vertical distance to Hydrology

This attribute had negative values, so it would be interesting to look at boxplots to see what they look like.


```python
plt.figure(figsize=(10,5))
ax = sns.boxplot(x='Wilderness_Area',y='vDistance_to_Hydrology',data=df, palette = "Blues_d")
ax.set_title('Elevation by Cover Type')
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3]), <a list of 4 Text xticklabel objects>)




![png](output_72_1.png)


The area with the lowest elevation, Cache la Poudre, and the least number of trees, also has the highest mean distance to water. The Comanche Peak area has the most outliers probably marking a very diverse terrain. Comanche Peak, Neota, and Rawah have about the same distance to hydrology. This may have something to do with the big body of water that sits between all of them.

[top](#toc)
<a id='Attribute_Attribute'></a>
## Attribute - Attribute Relationships


```python
pd.crosstab(df.Soil_Type, df.Wilderness_Area, margins=True, margins_name="Total")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Soil_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Soil Type 1</th>
      <td>3031</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>Soil Type 10</th>
      <td>17914</td>
      <td>14720</td>
      <td>0</td>
      <td>0</td>
      <td>32634</td>
    </tr>
    <tr>
      <th>Soil Type 11</th>
      <td>596</td>
      <td>11814</td>
      <td>0</td>
      <td>0</td>
      <td>12410</td>
    </tr>
    <tr>
      <th>Soil Type 12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29971</td>
      <td>29971</td>
    </tr>
    <tr>
      <th>Soil Type 13</th>
      <td>0</td>
      <td>17176</td>
      <td>255</td>
      <td>0</td>
      <td>17431</td>
    </tr>
    <tr>
      <th>Soil Type 14</th>
      <td>359</td>
      <td>240</td>
      <td>0</td>
      <td>0</td>
      <td>599</td>
    </tr>
    <tr>
      <th>Soil Type 15</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Soil Type 16</th>
      <td>263</td>
      <td>325</td>
      <td>117</td>
      <td>2140</td>
      <td>2845</td>
    </tr>
    <tr>
      <th>Soil Type 17</th>
      <td>793</td>
      <td>2629</td>
      <td>0</td>
      <td>0</td>
      <td>3422</td>
    </tr>
    <tr>
      <th>Soil Type 18</th>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>1829</td>
      <td>1899</td>
    </tr>
    <tr>
      <th>Soil Type 19</th>
      <td>0</td>
      <td>675</td>
      <td>597</td>
      <td>2749</td>
      <td>4021</td>
    </tr>
    <tr>
      <th>Soil Type 2</th>
      <td>2144</td>
      <td>5381</td>
      <td>0</td>
      <td>0</td>
      <td>7525</td>
    </tr>
    <tr>
      <th>Soil Type 20</th>
      <td>0</td>
      <td>2452</td>
      <td>55</td>
      <td>6752</td>
      <td>9259</td>
    </tr>
    <tr>
      <th>Soil Type 21</th>
      <td>0</td>
      <td>838</td>
      <td>0</td>
      <td>0</td>
      <td>838</td>
    </tr>
    <tr>
      <th>Soil Type 22</th>
      <td>0</td>
      <td>8362</td>
      <td>5363</td>
      <td>19648</td>
      <td>33373</td>
    </tr>
    <tr>
      <th>Soil Type 23</th>
      <td>0</td>
      <td>21071</td>
      <td>8153</td>
      <td>28528</td>
      <td>57752</td>
    </tr>
    <tr>
      <th>Soil Type 24</th>
      <td>0</td>
      <td>16252</td>
      <td>2123</td>
      <td>2903</td>
      <td>21278</td>
    </tr>
    <tr>
      <th>Soil Type 25</th>
      <td>0</td>
      <td>0</td>
      <td>474</td>
      <td>0</td>
      <td>474</td>
    </tr>
    <tr>
      <th>Soil Type 26</th>
      <td>0</td>
      <td>2589</td>
      <td>0</td>
      <td>0</td>
      <td>2589</td>
    </tr>
    <tr>
      <th>Soil Type 27</th>
      <td>0</td>
      <td>1086</td>
      <td>0</td>
      <td>0</td>
      <td>1086</td>
    </tr>
    <tr>
      <th>Soil Type 28</th>
      <td>0</td>
      <td>946</td>
      <td>0</td>
      <td>0</td>
      <td>946</td>
    </tr>
    <tr>
      <th>Soil Type 29</th>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>115173</td>
      <td>115247</td>
    </tr>
    <tr>
      <th>Soil Type 3</th>
      <td>2455</td>
      <td>2368</td>
      <td>0</td>
      <td>0</td>
      <td>4823</td>
    </tr>
    <tr>
      <th>Soil Type 30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30170</td>
      <td>30170</td>
    </tr>
    <tr>
      <th>Soil Type 31</th>
      <td>0</td>
      <td>25240</td>
      <td>426</td>
      <td>0</td>
      <td>25666</td>
    </tr>
    <tr>
      <th>Soil Type 32</th>
      <td>0</td>
      <td>48758</td>
      <td>3761</td>
      <td>0</td>
      <td>52519</td>
    </tr>
    <tr>
      <th>Soil Type 33</th>
      <td>0</td>
      <td>42337</td>
      <td>2817</td>
      <td>0</td>
      <td>45154</td>
    </tr>
    <tr>
      <th>Soil Type 34</th>
      <td>0</td>
      <td>1611</td>
      <td>0</td>
      <td>0</td>
      <td>1611</td>
    </tr>
    <tr>
      <th>Soil Type 35</th>
      <td>0</td>
      <td>732</td>
      <td>503</td>
      <td>656</td>
      <td>1891</td>
    </tr>
    <tr>
      <th>Soil Type 36</th>
      <td>0</td>
      <td>119</td>
      <td>0</td>
      <td>0</td>
      <td>119</td>
    </tr>
    <tr>
      <th>Soil Type 37</th>
      <td>0</td>
      <td>66</td>
      <td>0</td>
      <td>232</td>
      <td>298</td>
    </tr>
    <tr>
      <th>Soil Type 38</th>
      <td>0</td>
      <td>5993</td>
      <td>2073</td>
      <td>7507</td>
      <td>15573</td>
    </tr>
    <tr>
      <th>Soil Type 39</th>
      <td>0</td>
      <td>6117</td>
      <td>931</td>
      <td>6758</td>
      <td>13806</td>
    </tr>
    <tr>
      <th>Soil Type 4</th>
      <td>1238</td>
      <td>11158</td>
      <td>0</td>
      <td>0</td>
      <td>12396</td>
    </tr>
    <tr>
      <th>Soil Type 40</th>
      <td>0</td>
      <td>2309</td>
      <td>2092</td>
      <td>4349</td>
      <td>8750</td>
    </tr>
    <tr>
      <th>Soil Type 5</th>
      <td>1597</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1597</td>
    </tr>
    <tr>
      <th>Soil Type 6</th>
      <td>6575</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6575</td>
    </tr>
    <tr>
      <th>Soil Type 7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>Soil Type 8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>179</td>
      <td>179</td>
    </tr>
    <tr>
      <th>Soil Type 9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1147</td>
      <td>1147</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>36968</td>
      <td>253364</td>
      <td>29884</td>
      <td>260796</td>
      <td>581012</td>
    </tr>
  </tbody>
</table>
</div>



**Soil Type 1** and **15** are only present in the **Cache_la_Poudre** wilderness area.  **Soil Type 7**, **8**, **9**, **12**, and **30**,  are only present in the **Rawah** wilderness area.  **Soil Type 14**, **17**, **2**, **3**, and **4**,  are only present in the **Cache_la_Poudre** and **Comanche_Peak** wilderness area.  Only **Soil Type 16** is present in all wilderness areas.


```python
pd.crosstab(df.Soil_Type, df.Wilderness_Area, margins=True, margins_name="Total", normalize='index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Wilderness_Area</th>
      <th>Cache_la_Poudre</th>
      <th>Comanche_Peak</th>
      <th>Neota</th>
      <th>Rawah</th>
    </tr>
    <tr>
      <th>Soil_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Soil Type 1</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 10</th>
      <td>0.548937</td>
      <td>0.451063</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 11</th>
      <td>0.048026</td>
      <td>0.951974</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 12</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Soil Type 13</th>
      <td>0.000000</td>
      <td>0.985371</td>
      <td>0.014629</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 14</th>
      <td>0.599332</td>
      <td>0.400668</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 15</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 16</th>
      <td>0.092443</td>
      <td>0.114236</td>
      <td>0.041125</td>
      <td>0.752197</td>
    </tr>
    <tr>
      <th>Soil Type 17</th>
      <td>0.231736</td>
      <td>0.768264</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 18</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.036862</td>
      <td>0.963138</td>
    </tr>
    <tr>
      <th>Soil Type 19</th>
      <td>0.000000</td>
      <td>0.167869</td>
      <td>0.148471</td>
      <td>0.683661</td>
    </tr>
    <tr>
      <th>Soil Type 2</th>
      <td>0.284917</td>
      <td>0.715083</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 20</th>
      <td>0.000000</td>
      <td>0.264823</td>
      <td>0.005940</td>
      <td>0.729236</td>
    </tr>
    <tr>
      <th>Soil Type 21</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 22</th>
      <td>0.000000</td>
      <td>0.250562</td>
      <td>0.160699</td>
      <td>0.588739</td>
    </tr>
    <tr>
      <th>Soil Type 23</th>
      <td>0.000000</td>
      <td>0.364853</td>
      <td>0.141173</td>
      <td>0.493974</td>
    </tr>
    <tr>
      <th>Soil Type 24</th>
      <td>0.000000</td>
      <td>0.763794</td>
      <td>0.099774</td>
      <td>0.136432</td>
    </tr>
    <tr>
      <th>Soil Type 25</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 26</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 27</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 28</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 29</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000642</td>
      <td>0.999358</td>
    </tr>
    <tr>
      <th>Soil Type 3</th>
      <td>0.509019</td>
      <td>0.490981</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 30</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Soil Type 31</th>
      <td>0.000000</td>
      <td>0.983402</td>
      <td>0.016598</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 32</th>
      <td>0.000000</td>
      <td>0.928388</td>
      <td>0.071612</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 33</th>
      <td>0.000000</td>
      <td>0.937614</td>
      <td>0.062386</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 34</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 35</th>
      <td>0.000000</td>
      <td>0.387097</td>
      <td>0.265997</td>
      <td>0.346906</td>
    </tr>
    <tr>
      <th>Soil Type 36</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 37</th>
      <td>0.000000</td>
      <td>0.221477</td>
      <td>0.000000</td>
      <td>0.778523</td>
    </tr>
    <tr>
      <th>Soil Type 38</th>
      <td>0.000000</td>
      <td>0.384833</td>
      <td>0.133115</td>
      <td>0.482052</td>
    </tr>
    <tr>
      <th>Soil Type 39</th>
      <td>0.000000</td>
      <td>0.443068</td>
      <td>0.067434</td>
      <td>0.489497</td>
    </tr>
    <tr>
      <th>Soil Type 4</th>
      <td>0.099871</td>
      <td>0.900129</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 40</th>
      <td>0.000000</td>
      <td>0.263886</td>
      <td>0.239086</td>
      <td>0.497029</td>
    </tr>
    <tr>
      <th>Soil Type 5</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 6</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Soil Type 7</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Soil Type 8</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Soil Type 9</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.063627</td>
      <td>0.436074</td>
      <td>0.051434</td>
      <td>0.448865</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sample = df.sample(frac=0.02, replace=True, random_state=1) #sample with 10% of the data ~58,000 lines
df_sample.shape
```




    (11620, 15)




```python
sns.set(style="white")

g = sns.PairGrid(df_sample[quantitative], diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d") # use joint kde on the lower triangle
g.map_upper(plt.scatter) # scatter on the upper
g.map_diag(sns.kdeplot, lw=3) # kde histogram on the diagonal
```

    CPU times: user 5 µs, sys: 0 ns, total: 5 µs
    Wall time: 9.06 µs





    <seaborn.axisgrid.PairGrid at 0x7f2f74f89ba8>




![png](output_79_2.png)


We can see the bimodal nature of aspect and how it relates to the other continuous variables throug this plot. Also we can quickly see some positive and negative correlation for the hillshade index.


```python
%matplotlib inline
cvt = df[quantitative]
corr = cvt.corr()
plt.figure(figsize=[16,20])
sns.heatmap(corr,annot=True, fmt=".1", cmap="BrBG")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f880a019c18>




![png](output_81_1.png)


Very quickly we can see how the Hillshade variables are correlated with eachother.  **Hillshade_9am** and **Hillshade_3pm** have the highest correlation among all of the continuous variables (-.8).  We also see that **Hillshade_3pm** is also highly correlated with **Hillshade_Noon** (.6).  Outside of the HIllshade variables we can see that the **Aspect** variable is correlated with **Hillshade_3pm** (.6) and **Hillshade_Noon** (-.6).  The variables **Slope** and **Hillshade_Noon** also have a relatively high correlation (-.5).  The last set of variables worth mentioning are the distance to hydrology variables.  The vertical and horizontal **Distance_to_Hydrology** have a correlation coefficient of .6.

When considering feature selection we try to avoid including variables with high levels of correlation.  Based on the correlation analysis we would need to investigate strategies to combine the variables mentioned above to build a better predictive model.

[top](#toc)
<a id='Attribute_Response'></a>
## Attribute - Response Relationship


```python
df[quantitative + ['Cover_Type']].groupby(by='Cover_Type').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>hDistance_to_Hydrology</th>
      <th>vDistance_to_Hydrology</th>
      <th>hDistance_to_Roads</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>hDistance_to_Fire Points</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aspen</th>
      <td>2787.417571</td>
      <td>139.283051</td>
      <td>16.641315</td>
      <td>212.354893</td>
      <td>50.610344</td>
      <td>1349.765722</td>
      <td>223.474876</td>
      <td>219.035816</td>
      <td>121.920889</td>
      <td>1577.719794</td>
    </tr>
    <tr>
      <th>Cottonwood/Willow</th>
      <td>2223.939934</td>
      <td>137.139425</td>
      <td>18.528941</td>
      <td>106.934838</td>
      <td>41.186749</td>
      <td>914.199490</td>
      <td>228.345832</td>
      <td>216.997088</td>
      <td>111.392792</td>
      <td>859.124135</td>
    </tr>
    <tr>
      <th>Douglas-fir</th>
      <td>2419.181897</td>
      <td>180.539068</td>
      <td>19.048886</td>
      <td>159.853458</td>
      <td>45.437439</td>
      <td>1037.169805</td>
      <td>192.844302</td>
      <td>209.827662</td>
      <td>148.284044</td>
      <td>1055.351471</td>
    </tr>
    <tr>
      <th>Krummholz</th>
      <td>3361.928669</td>
      <td>153.236226</td>
      <td>14.255924</td>
      <td>356.994686</td>
      <td>69.474305</td>
      <td>2738.250463</td>
      <td>216.967723</td>
      <td>221.746026</td>
      <td>134.932033</td>
      <td>2070.031594</td>
    </tr>
    <tr>
      <th>Lodgepole Pine</th>
      <td>2920.936061</td>
      <td>152.060515</td>
      <td>13.550499</td>
      <td>279.916442</td>
      <td>45.884219</td>
      <td>2429.530799</td>
      <td>213.844423</td>
      <td>225.326596</td>
      <td>142.983466</td>
      <td>2168.154849</td>
    </tr>
    <tr>
      <th>Ponderosa Pine</th>
      <td>2394.509845</td>
      <td>176.372490</td>
      <td>20.770208</td>
      <td>210.276473</td>
      <td>62.446915</td>
      <td>943.940734</td>
      <td>201.918415</td>
      <td>215.826537</td>
      <td>140.367176</td>
      <td>910.955949</td>
    </tr>
    <tr>
      <th>Spruce/Fir</th>
      <td>3128.644888</td>
      <td>156.138227</td>
      <td>13.127110</td>
      <td>270.555245</td>
      <td>42.156939</td>
      <td>2614.834517</td>
      <td>211.998782</td>
      <td>223.430211</td>
      <td>143.875038</td>
      <td>2009.253517</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[quantitative + ['Cover_Type']].groupby(by='Cover_Type').median()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>hDistance_to_Hydrology</th>
      <th>vDistance_to_Hydrology</th>
      <th>hDistance_to_Roads</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>hDistance_to_Fire Points</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aspen</th>
      <td>2796</td>
      <td>111</td>
      <td>16</td>
      <td>175</td>
      <td>35</td>
      <td>1282</td>
      <td>228</td>
      <td>224</td>
      <td>128</td>
      <td>1471</td>
    </tr>
    <tr>
      <th>Cottonwood/Willow</th>
      <td>2231</td>
      <td>119</td>
      <td>19</td>
      <td>30</td>
      <td>6</td>
      <td>949</td>
      <td>235</td>
      <td>220</td>
      <td>113</td>
      <td>806</td>
    </tr>
    <tr>
      <th>Douglas-fir</th>
      <td>2428</td>
      <td>173</td>
      <td>19</td>
      <td>134</td>
      <td>34</td>
      <td>966</td>
      <td>196</td>
      <td>213</td>
      <td>150</td>
      <td>942</td>
    </tr>
    <tr>
      <th>Krummholz</th>
      <td>3363</td>
      <td>123</td>
      <td>13</td>
      <td>283</td>
      <td>43</td>
      <td>2654</td>
      <td>221</td>
      <td>224</td>
      <td>140</td>
      <td>1969</td>
    </tr>
    <tr>
      <th>Lodgepole Pine</th>
      <td>2935</td>
      <td>127</td>
      <td>13</td>
      <td>240</td>
      <td>30</td>
      <td>2039</td>
      <td>219</td>
      <td>227</td>
      <td>142</td>
      <td>1846</td>
    </tr>
    <tr>
      <th>Ponderosa Pine</th>
      <td>2404</td>
      <td>160</td>
      <td>21</td>
      <td>190</td>
      <td>50</td>
      <td>853</td>
      <td>213</td>
      <td>221</td>
      <td>142</td>
      <td>824</td>
    </tr>
    <tr>
      <th>Spruce/Fir</th>
      <td>3146</td>
      <td>122</td>
      <td>12</td>
      <td>218</td>
      <td>24</td>
      <td>2389</td>
      <td>216</td>
      <td>226</td>
      <td>144</td>
      <td>1825</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig,axs=plt.subplots(6, figsize=(10,25))
sns.boxplot(x='Cover_Type',y='Elevation',data=df,ax=axs[0], palette = "Blues_d")#highest in 1 & 7 lowest in 4
sns.boxplot(x='Cover_Type',y='Aspect',data=df,ax=axs[1], palette = "Blues_d")
sns.boxplot(x='Cover_Type',y='Slope',data=df,ax=axs[2], palette = "Blues_d")
sns.boxplot(x='Cover_Type',y='hDistance_to_Hydrology',data=df,ax=axs[3], palette = "Blues_d")
sns.boxplot(x='Cover_Type',y='vDistance_to_Hydrology',data=df,ax=axs[4], palette = "Blues_d")
sns.boxplot(x='Cover_Type',y='hDistance_to_Roads',data=df,ax=axs[5], palette = "Blues_d")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb5d86dccc0>




![png](output_86_1.png)


## Analysis of Continuous Variables within each Cover Type
These boxplots help visualize the distribution of our continuous variables within each cover type.  Here You'll see that the majority of the variables dont have much of a distinct distribution when comparing between the cover types.  The exceptions seem to be the variables **Elevation**, **Slope**, **Horizontal Distance to Hydrology**, **Horizontal Distance to Roadways**, and **Horizontal Distance to Fire Points**.  This analysis can give us an idea of which variables to include when building a predictive model.

If there was one variable to highlight, it would be elevation.  The degree of distinction among between the cover types is quite easy to see from the box plots.


```python
fig,axs=plt.subplots(5, figsize=(9,60))
sns.violinplot(x='Cover_Type',y='Elevation',data=df,ax=axs[0])
sns.violinplot(x='Cover_Type',y='Aspect',data=df,ax=axs[1])
sns.violinplot(x='Cover_Type',y='Slope',data=df,ax=axs[2])
sns.violinplot(x='Cover_Type',y='hDistance_to_Hydrology',data=df)
sns.violinplot(x='Cover_Type',y='vDistance_to_Hydrology',data=df)
sns.violinplot(x='Cover_Type',y='hDistance_to_Roads',data=df,ax=axs[3])
for ax in axs:
    plt.sca(ax)
    plt.xticks(rotation=60)
```


![png](output_88_0.png)


[top](#toc)
<a id='Additional_Features'></a>
## Additional Features


```python
df["Distance_to_Hydrology"] = ( (df["hDistance_to_Hydrology"] ** 2) + \
                               (df["vDistance_to_Hydrology"] ** 2) ) ** (0.5)

plt.figure(figsize=(10,5))
ax = sns.boxplot(x='Cover_Type',y='Distance_to_Hydrology',data=df, palette = "Blues_d")
ax.set_title('Distance_to_Hydrology')
plt.xticks(rotation=60)
#sns.swarmplot(x='Cover_Type', y="Distance_To_Hydrology", data=df, color=".25")
```




    (array([0, 1, 2, 3, 4, 5, 6]), <a list of 7 Text xticklabel objects>)




![png](output_90_1.png)


When reviewing our correlation analysis we observed that the **Distance_to_Hydrology** variables were highly correlated.  To help reduce the number of correlated variables we can find ways to combine certain data features.  Using the Pythagorean Theorem we can determine the straight distance to hydrology and create 1 variable that incorporates two highly correlated variables.

[top](#toc)
#### end
