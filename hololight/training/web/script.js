// Define study
const study = lab.util.fromObject({
  "title": "root",
  "type": "lab.flow.Sequence",
  "parameters": {},
  "plugins": [
    {
      "type": "lab.plugins.Metadata",
      "path": undefined
    },
    {
      "type": "lab.plugins.Download",
      "filePrefix": "go-task",
      "path": undefined
    }
  ],
  "metadata": {
    "title": "Go Task",
    "description": "Prompt the subject to perform a mental feat periodically.",
    "repository": "",
    "contributors": "Griffin Milsap \u003Cgriffin.milsap@jhuapl.edu\u003E"
  },
  "files": {},
  "responses": {},
  "content": [
    {
      "type": "lab.html.Page",
      "items": [
        {
          "type": "text",
          "title": "\"GO\" Task",
          "content": "Stare at the fixation cross throughout the experiment.  When the circle appears around the cross, perform a mental feat (try imagining squeezing your fists and wiggling your fingers) until the circle disappears.  Continue until the task ends."
        }
      ],
      "scrollTop": true,
      "submitButtonText": "Continue →",
      "submitButtonPosition": "right",
      "files": {},
      "responses": {
        "": ""
      },
      "parameters": {},
      "messageHandlers": {
        "after:end": function anonymous(
) {
window.socket = new WebSocket( "wss://" + location.hostname + ':5545' );   
window.socket.onopen = () => console.log( 'Input Socket Connected' );
  
window.socket.onmessage = ( msg ) => {
  var content = JSON.parse( msg.data );
  for( var key in content )
    this.state[key] = content[key];
  this.commit();
};

this.internals.controller.datastore.on( 
  'commit', () => {
    if( window.socket )
      if( window.socket.readyState === window.socket.OPEN ) {
        window.socket.send( JSON.stringify( this.state ) ) 
      }
  } 
)
}
      },
      "title": "Instructions"
    },
    {
      "type": "lab.html.Frame",
      "context": "\u003Cmain data-labjs-section=\"frame\"\u003E\n  \u003C!-- Content gets inserted here --\u003E\n\u003C\u002Fmain\u003E",
      "contextSelector": "[data-labjs-section=\"frame\"]",
      "files": {},
      "responses": {
        "": ""
      },
      "parameters": {},
      "messageHandlers": {},
      "title": "Task Frame",
      "content": {
        "type": "lab.flow.Loop",
        "templateParameters": [
          {
            "condition": "1",
            "": ""
          }
        ],
        "sample": {
          "mode": "draw-shuffle",
          "n": "30"
        },
        "files": {},
        "responses": {
          "": ""
        },
        "parameters": {},
        "messageHandlers": {},
        "title": "Trial Loop",
        "shuffleGroups": [],
        "template": {
          "type": "lab.flow.Sequence",
          "files": {},
          "responses": {
            "": ""
          },
          "parameters": {},
          "messageHandlers": {},
          "title": "Trial",
          "content": [
            {
              "type": "lab.canvas.Screen",
              "content": [
                {
                  "type": "rect",
                  "left": 0,
                  "top": 0,
                  "angle": 0,
                  "width": "10",
                  "height": 50,
                  "stroke": null,
                  "strokeWidth": 1,
                  "fill": "black"
                },
                {
                  "type": "rect",
                  "left": 0,
                  "top": 0,
                  "angle": 0,
                  "width": 50,
                  "height": "10",
                  "stroke": null,
                  "strokeWidth": 1,
                  "fill": "black"
                }
              ],
              "viewport": [
                800,
                600
              ],
              "files": {},
              "responses": {
                "": ""
              },
              "parameters": {},
              "messageHandlers": {},
              "title": "Fixation",
              "timeout": "5000"
            },
            {
              "type": "lab.canvas.Screen",
              "content": [
                {
                  "type": "rect",
                  "left": 0,
                  "top": 0,
                  "angle": 0,
                  "width": "10",
                  "height": 50,
                  "stroke": null,
                  "strokeWidth": 1,
                  "fill": "black"
                },
                {
                  "type": "rect",
                  "left": 0,
                  "top": 0,
                  "angle": 0,
                  "width": 50,
                  "height": "10",
                  "stroke": null,
                  "strokeWidth": 1,
                  "fill": "black"
                },
                {
                  "type": "circle",
                  "left": 0,
                  "top": 0,
                  "angle": 0,
                  "width": "100",
                  "height": 55,
                  "stroke": "#000000",
                  "strokeWidth": 10,
                  "fill": ""
                }
              ],
              "viewport": [
                800,
                600
              ],
              "files": {},
              "responses": {
                "": ""
              },
              "parameters": {},
              "messageHandlers": {},
              "title": "Go",
              "timeout": "3000"
            }
          ]
        }
      }
    }
  ]
})

// Let's go!
study.run()