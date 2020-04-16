var express = require('express');
//var client = require('./client')
var app = express();
var morgan = require('morgan')

app.use(morgan("dev"));

app.engine('html', require('ejs').__express);
app.set('view engine', 'html');
app.use(express.static(__dirname + '/public'));
app.set('view engine', 'ejs');



function getJSONData(latitude, longitude, radius) {
  // Within a radius of this point, return the heat map results
  //var coords = JSON.parse(dummyjsonMap);
  var coords = dummyMap.coordinates;
  var selected = [];
  for (c in coords) {
    if (c.latitude < latitude + radius && 
      c.latitude > latitude - radius &&
      c.longitude < longitude + radius &&
      c.longitude > longitude - radius) {
        selected.push(c);
      }
  }

  return selected;
}


// TODO: this should be a JSON object input not a regular object
var dummyMap =
{
	"coordinates": [
		{
			"latitude": "40.416775",
			"longitude": "-3.70379",
			"color": "GREEN",
			"weight": "6"
		},
		{
			"latitude": "41.385064",
			"longitude": "2.173403",
			"color": "GREEN",
			"weight": "2"
		},
		{
			"latitude": "52.130661",
			"longitude": "-3.783712",
			"color": "GREEN",
			"weight": "2"
		},
		{
			"latitude": "55.378051",
			"longitude": "-3.435973",
			"color": "GREEN",
			"weight": "8"
		},
		{
			"latitude": "-40.900557",
			"longitude": "-174.885971",
			"color": "GREEN",
			"weight": "6"
		},
		{
			"latitude": "40.714353",
			"longitude": "-74.005973",
			"color": "RED",
			"weight": "6"
		}
	]
};


app.get('/', function (req, res) {
  res.render('sapling.ejs.html');
});


app.get('/getRegion', function (req, res) {
  //client.getRegionFunction();
  console.log("Reached");
  //Take point, radius. Return "map overlay data"
  //latitude = req.param("LATITUDE");
  //longitude = req.param("LONGITUDE");
 // radius = req.param("RADIUS");
 latitude = 20;
 longitude = 100;
 radius = 500;
  // Call some function here on point and radius to return JSON data
  //jsonData = getJSONData(latitude, longitude, radius);
  var coords = [];
  var leftbottomlat = latitude - radius
  var leftbottomlong = longitude - radius;
  var rightbottomlat = latitude + radius;
  var rightbottomlong = latitude - radius;
  var lefttoplat = latitude - radius;
  var lefttoplong = latitude + radius;
  var righttoplat = latitude + radius;
  var righttoplong = longitude + radius;
  res.render('map.html');
  /*for (d in jsonData) {
    coords += { location: new google.maps.LatLng(d.latitude, d.longitude), weight: d.weight }
  }*/

  /*document.getElementById("leftbottomlat").innerHTML = leftbottomlat;
  document.getElementById("leftbottomlong").innerHTML = leftbottomlong;
  document.getElementById("rightbottomlat").innerHTML = rightbottomlat;
  document.getElementById("rightbottomlong").innerHTML = rightbottomlong;
  document.getElementById("lefttoplat").innerHTML = lefttoplat;
  document.getElementById("lefttoplong").innerHTML = lefttoplong;
  document.getElementById("righttoplat").innerHTML = righttoplat;
  document.getElementById("righttoplong").innerHTML = righttoplong;*/
  //document.getElementById("coords").innerHTML = coords;
  //res.json(getJSONData(latitude, longitude, radius));
});

app.listen(3000);


//module.exports = app;