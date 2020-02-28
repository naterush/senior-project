var express = require('express');
var app = express();

app.engine('html', require('ejs').__express);
app.set('view engine', 'html');
app.use(express.static(__dirname + '/public'));
app.set('view engine', 'ejs');


dummyjsonFile = {
  "glossary": {
      "title": "example glossary",
  "GlossDiv": {
          "title": "S",
    "GlossList": {
              "GlossEntry": {
                  "ID": "SGML",
        "SortAs": "SGML",
        "GlossTerm": "Standard Generalized Markup Language",
        "Acronym": "SGML",
        "Abbrev": "ISO 8879:1986",
        "GlossDef": {
                      "para": "A meta-markup language, used to create markup languages such as DocBook.",
          "GlossSeeAlso": ["GML", "XML"]
                  },
        "GlossSee": "markup"
              }
          }
      }
  }
};



function getJSONData(point, radius) {
  return dummyjsonFile;
}

app.get('/', function (req, res) {
  res.render('map.html');
});

app.get('/sapling', function (req, res) {
  res.render('sapling.ejs.html');
});

app.get('/getRegion', function (req, res) {
  //Take point, radius. Return "map overlay data"
  point = req.param("POINT");
  radius = req.param("RADIUS");
  // Call some function here on point and radius to return JSON data
  res.json(getJSONData(point, radius));
});

app.listen(3000);


//module.exports = app;