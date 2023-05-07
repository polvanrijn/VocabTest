/// Based on code at http://bl.ocks.org/fabiovalse/dfcd8104a79aed092af1
///  Modified by Andrew Su on 2016-06-11 

var width = 5500,    // svg width
    height = 5500;   // svg height

// set globalTextColor to:
//    - a hex color value for a single text color throughout
//    - null for a background-dependent choice of white/black
var globalTextColor = "#000000";

var rotation = 2 * Math.PI;   // global variable

var spiralParams = {
  "level1": { 
    "opacity": 1,
    "r": 70,
    "chord": 150,
    "awayStep": 50,
    "fontSize": 50,
    "strokewidth": 25,
    "nodeColor": "#379154"
  },
  "level2": { 
    "opacity": 1,
    "r": 62,
    "chord": 75,
    "awayStep": 35,
    "fontSize": 50,
    "strokewidth": 15,
  },
  "level3": { 
    "opacity": .5,
    "r": 25,
    "chord": 50,
    "awayStep": 25,
    "fontSize": 0,
    "strokewidth": 7,
  },
  "level4": { 
    "opacity": .3,
    "r": 15,
    "chord": 25,
    "awayStep": 25,
    "fontSize": 0,
    "strokewidth": 3,
  },
  "level5": { 
    "opacity": .15,
    "r": 10,
    "chord": 25,
    "awayStep": 25,
    "fontSize": 0,
    "strokewidth": 3,
  },
  "default": { 
    "opacity": .1,
    "r": 10,
    "chord": 30,
    "awayStep": 50,
    "fontSize": 0,
    "strokewidth": 3,
  },
  "defaultColor": "#A9A9A9"   // darkgray
}

// SPIRAL
// colors based on https://color.adobe.com/custom-pastels-color-theme-1993286/?showPublished=true
var colorScale = d3.scale.linear()
    .range([ '#379154', '#39B4BF', '#FFE666', '#946FB0', '#E54E67' ]) // or use hex values
    .domain([1984, 1992, 2000, 2008, 2016]);

// removing yellow
/*var colorScale = d3.scale.linear()
    .range([ '#379154', '#39B4BF', '#946FB0', '#E54E67' ]) // or use hex values
    .domain([1980, 1992, 2004, 2016]);
*/
/*
// SPIRAL2
// colors based on tweaked version of https://color.adobe.com/custom-pastels-color-theme-1993286/?showPublished=true
var colorScale = d3.scale.linear()
    .range([ '#379154', '#39B4BF', '#FFE666', '#6A5B7D', '#FF8469' ]) // or use hex values
    .domain([1980, 1989, 1998, 2007, 2016]);

// SPIRAL4
// colors based on https://color.adobe.com/Analogous-Water-CB-color-theme-8326220/
var colorScale = d3.scale.linear()
    .range([ '#5FDEDA', '#0098C7', '#004382', '#000C36' ]) // or use hex values
    .domain([1980, 1992, 2004, 2016]);
*/
// test case for background-dependent text color
/*
var colorScale = d3.scale.linear()
    .range([ '#000000', '#FFFFFF' ]) // or use hex values
    .domain([1980, 2016]);
*/

// hexToRgb from http://stackoverflow.com/questions/5623838/rgb-to-hex-and-hex-to-rgb
function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// calcTextColor adapted from http://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
function calcTextColor ( bgColor ) {
  var rgb = hexToRgb( bgColor );
  var contrast = (Math.round(rgb.r * 299) + Math.round(rgb.g * 587) + Math.round(rgb.b * 114)) / 1000;
  return (contrast >= 128) ? 'black' : 'white';
}

function calcCoords ( thisCenterX, thisCenterY, thisNumPoints, thisChord, thisAwayStep ) {
  
  var coords = [];    // output array
  var theta = thisChord / thisAwayStep;

  for ( i = 0; i < thisNumPoints; i++ ) {
      var away = thisAwayStep * theta;
      var around = theta + rotation;
    
      var x = thisCenterX + Math.cos ( around ) * away;
      var y = thisCenterY + Math.sin ( around ) * away;

      theta += thisChord / away;
    
      coords.push({x: x, y: y});
  }

  return coords;
}

var lineFunction = d3.svg.line()
                      .x(function(d) { return d.x; })
                      .y(function(d) { return d.y; })
                      .interpolate("cardinal");

function plotSpiral ( root, level ) {

  if( root.children.length == 0 ) { return; }

  // plot the children first, so they show up in back
  for( var i = 0; i < root.children.length; i++ ) {
    plotSpiral(root.children[i], level+1);
  }

  var tempArray = JSON.parse(JSON.stringify(root.children));   // so add parent coordinates to spiral
  tempArray.unshift(root);

  var strokewidth;
  if ( spiralParams["level".concat(level)].strokewidth === undefined ) {
    strokewidth = spiralParams["default"].strokewidth;
  } else {
    strokewidth = spiralParams["level".concat(level)].strokewidth;
  }

  svg.append("path")
    .attr("d", lineFunction(tempArray))
    .attr("stroke", "#F0986C")
    .attr("stroke-width", strokewidth )
    .attr("fill", "none");

  var circles = svg.selectAll("svg")  // changed this selector to "svg" and it works, not sure why
        .data(tempArray)
      .enter()
        .append("g")
        .append("svg:a")
          .attr("xlink:href", function(d) { return d.url; })
          .attr("xlink:show","new")

  circles.append("circle")
        .attr("cx", function (d) { return d.x; })
        .attr("cy", function (d) { return d.y; })
        .attr("r", function (d) { 
          var r; 
          var level = "level".concat(d.level);
          try{ r = spiralParams[level].r; }
          catch (err) { r = spiralParams["default"].r; }
          return r;
        })
        .style("opacity", function (d) {
          var opacity;
          var level = "level".concat(d.level);
          try{ opacity = spiralParams[level].opacity }
          catch (err) { opacity = spiralParams["default"].opacity }
          return opacity;
        })
        /* UNCOMMENT THE NEXT SECTION TO COLOR BY YEAR; COMMENT TO COLOR BY PERSON TYPE*/
        .style("fill", function (d) {
          var col;

          // if nodeColor exists for the level in spiralParams, use it
          var level = "level".concat(d.level);
          try{ col = spiralParams[level].nodeColor; }
          catch (err) {}

          // else, define color by the year (or default)
          if( col === undefined ) {
            var year = d.startYear;
            if( year === undefined || year === null ) {
              col = spiralParams["defaultColor"];
            } else {
              col = colorScale(year); 
            }
            if( col === "#NaNNaNNaN" ) { col = spiralParams["defaultColor"];}
          }
          d.bgColor = col;
          return col;
        })
        .attr("class", function (d) { return "level".concat(d.level).concat(" ").concat(d.type)});

  circles.append("text")
    .attr("text-anchor","middle")
    .attr("x", function(d){return d.x})
    .attr("y", function(d){return d.y})
    .attr("dy", "0.4em")
    .attr("font-size",function(d) {
      var fontSize;
      var level = "level".concat(d.level);
      try{ fontSize = spiralParams[level].fontSize }
      catch (err) { fontSize = spiralParams["default"].fontSize }
      return fontSize;
    })
    .attr("font-family","sans-serif")
    .style("fill",function(d) {
      if ( globalTextColor != null ) {
        return globalTextColor;
      } else {
        var textColor = calcTextColor( d.bgColor )
        return(textColor)
      }
    })
    .text(function(d) { return(d.name)});

}

function addCoords( root, centerX, centerY ) {
  root.x = centerX;
  root.y = centerY;

  var level = "level".concat(root.level);
  var chord;
  var awayStep;
  try {
    chord = spiralParams[level].chord,        // spacing between points
    awayStep = spiralParams[level].awayStep;  // rate of increase of spiral radius
  }
  catch (err) {
    chord = spiralParams["default"].chord;
    awayStep = spiralParams["default"].awayStep;
  }

  var coords = calcCoords(centerX, centerY, root.children.length, chord, awayStep )

  for(var i = 0; i<root.children.length; i++ ) {
    root.children[i] = addCoords( root.children[i], coords[i].x, coords[i].y )
  }

  return(root)
}

/* *****  MAIN starts here ***** */

var svg = d3.select("#chart").append("svg")
    .attr("width", width)
    .attr("height", height)
  .append("g");

var new_time;  // make this a global variable so I can inspect it

var centerX = width/2,   // x coordinate of spiral center
    centerY = height/2,  // y coordinate of spiral center
    chord = 30,          // spacing between points
    awayStep = 10;       // rate of increase of spiral radius

var z; 
d3.json("output_PGS.json", function(error, root) {
  if (error) throw error;

  root = addCoords( root, centerX, centerY );
  plotSpiral( root, 1 );

  z = root;   /// temporary so I can inspect it
});


