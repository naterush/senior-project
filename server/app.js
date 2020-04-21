var express = require('express');
const util = require('util');
var morgan = require('morgan');
const fs = require(`fs`).promises;
const exec = util.promisify(require('child_process').exec);
var app = express();

app.use(morgan("dev"));

app.engine('html', require('ejs').__express);
app.set('view engine', 'html');
app.use(express.static(__dirname + '/public'));
app.set('view engine', 'ejs');

app.use(express.urlencoded());

async function runModel(lat, long) {
  const { stdout, stderr } = await exec(`python3 main.py ${lat} ${long}`);
  console.log(stdout, stderr)

  if (stderr.length !== 0) {
    console.log(`Call to model with ${lat}, ${long} errored with :\n ${stderr}`);
    return {};
  }

  const outputArr = stdout.split("\n");
  const newfile = outputArr[outputArr.length - 2]

  // Read in the file
  const jsonData = await fs.readFile(newfile)

  return JSON.parse(jsonData);
}

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


app.get('/', async function (req, res) {
  res.render('sapling.ejs.html');
});

app.get('/map', async function (req, res) {
  res.render('map.html');
});

app.get('/getRegion', async function (req, res) {
  // We set a timeout of 15 minutes...
  // because the model might take a while
  req.setTimeout(15 * 60 * 1000);

  const {latitude, longitude} = req.query;

  console.log(`Got region ${latitude}, ${longitude}`);

  const modelJSONResult = await runModel(latitude, longitude);

  console.log(`Got region ${latitude}, ${longitude} : ${modelJSONResult}`);

  res.json(modelJSONResult);

  return

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

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});


//module.exports = app;