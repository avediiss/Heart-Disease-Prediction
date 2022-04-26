Highcharts.chart('container', {
  chart: {
    type: 'bar',
    marginTop: 60
  },
  title: {
    text: 'Heart Disease Deaths by Region'
  },
  xAxis: {
    categories: ['France', 'Italy', 'Germany', 'United Kingdom', 'Russia', 'United States'],
  },
  yAxis: {
    plotBands: [{
      color: 'rgba(56,116,168,.05)', // Color value
      from: 0, // Start of the plot band
      to: 119, // End of the plot band
      label: {
        text: 'Best', // Content of the label. 
        align: 'center', // Positioning of the label.
        verticalAlign: 'top',
        y: -6,
      }
    }, {
      color: 'rgba(56,116,168,.15)', // Color value
      from: 120, // Start of the plot band
      to: 177, // End of the plot band
      label: {
        text: 'Second Best', // Content of the label. 
        align: 'center', // Positioning of the label. 
        verticalAlign: 'top',
        y: -6,
      }
    }, {
      color: 'rgba(56,116,168,.3)', // Color value
      from: 178, // Start of the plot band
      to: 199, // End of the plot band
      label: {
        text: 'Medium', // Content of the label. 
        align: 'center', // Positioning of the label.
        verticalAlign: 'top',
        y: -6,
      }
    }, {
      color: 'rgba(56,116,168,.5)', // Color value
      from: 200, // Start of the plot band
      to: 230, // End of the plot band
      label: {
        text: 'Second Worst', // Content of the label. 
        align: 'center', // Positioning of the label.
        verticalAlign: 'top',
        y: -6,
      }
    }, {
      color: 'rgba(56,116,168,1)', // Color value
      from: 231, // Start of the plot band
      to: 524, // End of the plot band
      label: {
        text: 'Worst', // Content of the label. 
        align: 'center', // Positioning of the label.
        verticalAlign: 'top',
        y: -6,
      }
    }, ],
  },
  series: [{
    name: "title of series",
    data: [198.82, 286.32, 230.42, 232.55, 226.62, 175.48]
  }],
  credits: false,
});




//

var options = {
  chart: {
    type: 'pie'
  },
  series: [900, 100],
  labels: ["Male", "Female"],
  plotOptions: {
    pie: {
      customScale: 1.0,
      donut: {
        labels: {
          show: true
        }
      }
    }
  },
  theme: {
    mode: 'black',
    palette: 'palette'
  },
  stroke: {
    show: false,
    width: 0,
    colors: undefined
  },
  dataLabels: {
    enabled: false,
  },
  legend: {
    fontSize: '18px',
    horizontalAlign: 'center',
    position: 'top'
  },
  title: {
    text: 'Sex',
    align: 'center',
    style: {
      fontSize: '20px'
    }
  }
}

var options2 = {
  chart: {
    height: 400,
    type: 'bar',
  },
  plotOptions: {
    bar: {
      horizontal: false,
      columnWidth: '70%',
      endingShape: 'rounded'
    },
  },
  dataLabels: {
    enabled: false
  },
  stroke: {
    show: true,
    width: 2,
    colors: ['black']
  },
  series: [{
    name: 'probability',
    data: [150, 200, 500, 800, 600, 90, 200]
  }, {

  }],
  xaxis: {
    categories: ['20', '30', '40', '50', '60', '70', '80'],
  },
  yaxis: {

  },
  fill: {
    opacity: 8

  },
  theme: {
    mode: 'black',
    palette: 'palette'
  },
  title: {
    text: 'Age Distribution',
    align: 'center',
    style: {
      fontSize: '18px'
    }
  }
}


var chart = new ApexCharts(document.querySelector("#chart"), options);

chart.render();

var chart2 = new ApexCharts(document.querySelector("#chart2"), options2);

chart2.render();