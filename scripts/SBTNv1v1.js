////  08/24/2024 Natural Lands Map
// Mazur, E. et al. (2024) Natural Lands Map
// World Resources Institute & Science Basted Targets Network 

/// Use this asset: 
var naturalLands = ee.Image('WRI/SBTN/naturalLands/v1_1/2020');


/// For Visualization, use following script: 

// Create left panel 
var leftPanel = ui.Panel({
    layout: ui.Panel.Layout.flow('vertical'),
    style: {
        width: '400px'
    }
});

// Create map object
var mapPanel = ui.Map();

// Clear area and add elements
ui.root.clear();

//Split Panel - order matters
ui.root.add(ui.SplitPanel(leftPanel, mapPanel));

var origin = ee.Geometry.Point([0, 10])
mapPanel.centerObject(origin, 2)

var ASSET_FOLDER = "projects/wri-datalab/SBTN/natLands_v1_1_blended_assets"
var allClasses = ee.Image('projects/wri-datalab/SBTN/natLands_v1_1_blended_assets/classification_class_bands_v1_1')
var binary = ee.Image('projects/wri-datalab/SBTN/natLands_v1_1_blended_assets/binary_class_bands_v1_1')
var CLASS_COLORS = {
    'allClasses': ['fff', "246e24", "B9B91E", "6BAED6", "06A285", "fef3cc", "ACD1E8", "589558", "093d09", "dbdb7b", "99991a", "D3D3D3", "D3D3D4", "D3D3D5", "D3D3D6", "D3D3D7", "D3D3D8", "D3D3D9", "D3D3D1", "D3D3D2", "D3D2D3", ],
    'binary': ['969696', 'a8ddb5', 'fff'],
}

function viz_class_bands_blend(img, colors) {
    var intensity = img.reduce("sum").divide(255)
    var classes = img.toArray().arrayArgmax().arrayGet(0).updateMask(intensity.neq(0))

    var bands = img.rename(colors).pow(1.5)
    var viz = ee.Image(ee.List(colors).iterate(
        function(c, res) {
            return ee.Image(res).add(
                ee.Image(1).visualize({
                    'palette': [c]
                }).multiply(bands.select([c]))
            )
        }, ee.Image(0)
    )).divide(bands.mask(1).reduce("sum"))

    return viz.mask(intensity.multiply(2).pow(0.33))
}


mapPanel.addLayer(viz_class_bands_blend(allClasses, CLASS_COLORS['allClasses']), {
    'min': 0,
    'max': 255
}, 'Natural Lands - Classification', 1)
mapPanel.addLayer(viz_class_bands_blend(binary, CLASS_COLORS['binary']), {
    'min': 0,
    'max': 255
}, 'Natural Lands - Natural', 1)

// Country Boundaries
var geob = ee.FeatureCollection("projects/wri-datalab/geoBoundaries/geoBoundariesCGAZ_ADM0");
var empty = ee.Image().byte();
var outline = empty.paint({
    featureCollection: geob,
    // color: 1,
    width: 0.1
});
mapPanel.addLayer(outline, {
    palette: 'ffffff'
}, 'Country Boundaries'); // white




///////////////////////////////////////////////////////
/// PANEL - TEXT
///////////////////////////////////////////////////////


var colors = {
    'cyan': '#24C1E0',
    'transparent': '#11ffee00',
    'gray': '#F8F9FA'
};

var TITLE_STYLE = {
    fontSize: '32px',
    padding: '10px',
    color: '#616161',
    backgroundColor: colors.transparent,
};

var PARAGRAPH_STYLE = {
    fontSize: '14px',
    color: '#000',
    padding: '4px',
    backgroundColor: colors.transparent,
};

var LABEL_STYLE = {
    fontWeight: '50',
    textAlign: 'center',
    fontSize: '11px',
    backgroundColor: colors.transparent,
};




// adding a legend as a small rectangular panel in the Map view:
// set position of panel
var text1 = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px'
    }
});

// Create legend title
var textTitle1 = ui.Label({
    value: 'SBTN Natural Lands Map 2020 v1.1',
    style: TITLE_STYLE
});

// Add the title to the panel
text1.add(textTitle1);

// Create text chunk
var textContent0 = ui.Label({
    value: 'Mazur, E., Sims, M., et al., SBTN Natural Lands Map v1.1, 2025',
    style: PARAGRAPH_STYLE
});

// Add the text chunk to the panel
text1.add(textContent0);

// Create text chunk
var textContent1 = ui.Label({
    value: 'This app visualizes the SBTN Natural Lands Map, which is intended to be used in the Science Based Targets Network target on "no conversion of natural ecosystems." Data downloads are available on the GitHub.',
    style: PARAGRAPH_STYLE
});

// Add the text chunk to the panel
text1.add(textContent1);

var textContent1a = ui.Label({
    value: 'The "Natural Lands - Natural" layer shows the whole world divided into two classes: natural and non-natural according to the definitions found in the technical note.',
    style: PARAGRAPH_STYLE
});

// Add the text chunk to the panel
text1.add(textContent1a);


var textContent1c = ui.Label({
    value: 'The "Natural Lands - Classification" layer shows the natural areas labeled by land cover. See technical note here:',
    style: PARAGRAPH_STYLE
});

// Add the text chunk to the panel
text1.add(textContent1c);

// Create URL
var urlContent1 = ui.Label({
    value: 'Natural Lands Map v1.1 - Technical Note',
    style: PARAGRAPH_STYLE
}).setUrl('https://sciencebasedtargetsnetwork.org/wp-content/uploads/2025/02/Technical-Guidance-2025-Step3-Land-v1_1-Natural-Lands-Map.pdf');

// Add the url to the panel
text1.add(urlContent1);



// Create second url
var urlContent2 = ui.Label({
    value: 'Full SBTN Land Targets Guidance',
    style: PARAGRAPH_STYLE
}).setUrl('https://sciencebasedtargetsnetwork.org/wp-content/uploads/2024/09/Technical-Guidance-2024-Step3-Land-v1.pdf');

// Add the second url to the panel
text1.add(urlContent2);



// Create URL
var urlContent3 = ui.Label({
    value: 'GEE Asset: WRI/SBTN/naturalLands/v1_1/2020',
    style: PARAGRAPH_STYLE
}).setUrl('https://developers.google.com/earth-engine/datasets/catalog/WRI_SBTN_naturalLands_v1_1_2020');
// Add the url to the panel
text1.add(urlContent3);

// Create URL
var urlContent4 = ui.Label({
    value: 'GitHub',
    style: PARAGRAPH_STYLE
}).setUrl('https://github.com/wri/natural-lands-map');
// Add the url to the panel
text1.add(urlContent4);


// Create text
var textContent1d = ui.Label({
    value: 'This data product was developed as part of the broader Science Based Targets Network’s Land Hub in collaboration with WRI’s Land & Carbon Lab, the World Wildlife Fund and Systemiq.',
    style: PARAGRAPH_STYLE
});

// Add the text chunk to the panel
text1.add(textContent1d);



// Create URL
var urlContent5 = ui.Label({
    value: 'Contact us here with questions or comments',
    style: PARAGRAPH_STYLE
}).setUrl('https://survey.alchemer.com/s3/7998306/SBTN-Natural-Lands-Map-2020-v1-Feedback'); // change this to the GEE URL when it is live!

// Add the url to the panel
text1.add(urlContent5);



// add legend to left panel  
leftPanel.add(text1);



// Create and style 1st row of the legend.
var makeRow = function(color, name) {

    // Create the label that is the colored box:
    var colorBox = ui.Label({
        style: {
            backgroundColor: '#' + color,
            // Use padding to give the box height and width.
            padding: '8px',
            margin: '0 0 4px 6px'
        }
    });

    // Create the label filled with the description text:
    var description = ui.Label({
        value: name,
        style: {
            margin: '0 0 4px 6px',
            fontSize: '12px',
        }
    });

    // return the panel
    return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
    });
};


///////////////////////////////////////////////////////
/// LEGEND - BINARY
///////////////////////////////////////////////////////
var palette = ['a8ddb5', '969696']
var legend_colors = palette;
var legend_keys = ['Natural', 'Non-Natural']
// adding a legend as a small rectangular panel in the Map view:
// set position of panel
var legend = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px'
    }
});

// Create legend title
var legendTitle = ui.Label({
    value: 'Natural Lands - Natural',
    style: {
        fontWeight: 'bold',
        fontSize: '12px',
        margin: '0 0 4px 0',
        padding: '8px'
    }
});

// Add the title to the panel
legend.add(legendTitle);

//  Specify palette with the colors
var palette = legend_colors;

// name of the legend
var names = legend_keys;
// Add color and and names (i< should be less than number of classes)
for (var i = 0; i < 2; i++) {
    legend.add(makeRow(palette[i], names[i]));
}

// add legend to left panel  
// leftPanel.add(legend); 
// mapPanel.add(legend);



///////////////////////////////////////////////////////
/// LEGEND - NATURAL CLASSES
///////////////////////////////////////////////////////
var palette2 = [
    "246e24", // forest
    "589558", // wet forest
    "093d09", // peat forest
    "FFFFFF", // SPACE
    "B9B91E", // short veg
    "dbdb7b", // wet short veg
    "99991a", // peat short veg
    "FFFFFF", // SPACE
    "6BAED6", // water
    "06A285", // mangroves
    "fef3cc", // bare
    "ACD1E8", // snow
    "D3D3D3", // Mot Natural
]

var legend_colors2 = palette2;
var legend_keys2 = ['Forests', 'Wetland Forests', 'Peat Forests', ' ', 'Short Vegetation', 'Wetland Short Vegetation', 'Peat Short Vegetation', '', 'Water', 'Mangroves', 'Bare Ground', 'Snow', 'Not Natural']
// adding a legend as a small rectangular panel in the Map view:
// set position of panel
var legend2 = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px'
    }
});

// Create legend title
var legendTitle2 = ui.Label({
    value: 'Natural Lands - Classification',
    style: {
        fontWeight: 'bold',
        fontSize: '12px',
        margin: '0 0 4px 0',
        padding: '8px'
    }
});

// Add the title to the panel
legend2.add(legendTitle2);



//  Specify palette with the colors
var palette2 = legend_colors2;

// name of the legend
var names = legend_keys2;
// Add color and and names (i< should be less than number of classes)
for (var i = 0; i < 13; i++) {
    legend2.add(makeRow(palette2[i], names[i]));
}

// add legend to left panel  
// leftPanel.add(legend2); 
mapPanel.add(legend2);
mapPanel.add(legend);

// adding a legend as a small rectangular panel in the Map view:
// set position of panel
var text2 = ui.Panel({
    style: {
        position: 'bottom-left',
        padding: '8px 15px'
    }
});




///////////////////////////////////////////////////////
/// BASEMAP
///////////////////////////////////////////////////////
/////////// light basemap ///////////
var lightBasemap = [{
    featureType: 'administrative',
    elementType: 'all',
    stylers: [{
        color: '#5c5c5c'
    }, {
        visibility: 'off'
    }]
}, {
    featureType: 'administrative',
    elementType: 'labels.text.fill',
    stylers: [{
        visibility: 'off'
    }]
}, {
    featureType: 'landscape',
    elementType: 'all',
    stylers: [{
        color: '#e4e2dd'
    }, {
        visibility: 'on'
    }]
    // stylers: [{color: '#2b2b2b'}, {visibility: 'on'}]
}, {
    featureType: 'poi',
    elementType: 'all',
    stylers: [{
        visibility: 'off'
    }]
}, {
    featureType: 'road',
    elementType: 'all',
    stylers: [{
        visibility: 'off'
    }]
}, {
    featureType: 'water',
    elementType: 'all',
    stylers: [{
        color: '#ffffff'
    }, {
        visibility: 'on'
    }]
    // stylers: [{color: '#6BAED6'}, {visibility: 'on'}]
}];


mapPanel.setOptions(
    'lightBasemap', {
        lightBasemap: lightBasemap
    });