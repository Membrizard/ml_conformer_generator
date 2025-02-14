import { Streamlit, RenderData } from "streamlit-component-lib"

const speckRenderer = require('./renderer.js');
const speckSystem = require('./system.js');
const speckView = require('./view.js');
const speckInteractions = require('./interactions.js');
const speckColors = require('./colors.js');


let system = speckSystem.new();
let view = speckView.new();
view.resolution.x = 200;
view.resolution.y = 200;
view.bonds = true;
view.atomScale = 0.24;
view.relativeAtomScale = 0.64;
view.bondScale = 0.5;
view.brightness = 0.5;
view.outline = 0.0;
view.spf = 32;
view.bondThreshold = 1.2;
view.bondShade = 0.5;
view.atomShade = 0.5;
view.dofStrength = 0.0;
view.dofPosition = 0.5;

var renderer: any = null;
let needReset = false;
let current_schema = "speck";


let container = document.createElement("div");
let canvas = document.createElement('canvas')
canvas.addEventListener('dblclick', function () {
    center();
});
let topbar = document.createElement('div')
topbar.style.top = "0px"
topbar.style.height = "30px"
topbar.style.right = "0px"
topbar.style.position = "absolute"
topbar.style.background = "rgb(14, 17, 23)"
topbar.style.flexDirection = "row";
topbar.style.alignContent = "flex-end";
topbar.style.display = "flex";


let infoc = document.createElement("div");
infoc.style.fontSize = "10px";
infoc.style.color = "#AAA";
// infoc.innerHTML = 'Colors: ' + current_schema;



let autoscale = document.createElementNS("http://www.w3.org/2000/svg", "svg");
autoscale.setAttribute('width', "16");
autoscale.setAttribute('height', "16");
autoscale.setAttribute('viewBox', "0 0 16 16");
autoscale.setAttribute('fill', "#AAAAAA");
autoscale.addEventListener('mouseover', function () {
    autoscale.setAttribute('fill', "#666666");
});
autoscale.addEventListener('mouseout', function () {
    autoscale.setAttribute('fill', "#AAAAAA");
});
autoscale.innerHTML = '<g><path d="M 1 5 v -4 h 4 M 15 5 v -4 h -4 M 1 11 v 4 h 4 M 11 15 h 4 v -4 M 5 8 l 3 -3 l 3 3 l -3 3 l -3 -3"></path></g>';
autoscale.addEventListener('click', function () {
    center();
});

let autoscalec = document.createElement("div");
autoscalec.style.padding = "2px"
autoscalec.append(autoscale)

let camera = document.createElementNS("http://www.w3.org/2000/svg", "svg");
camera.setAttribute('width', "16");
camera.setAttribute('height', "16");
camera.setAttribute('viewBox', "0 0 16 16");
camera.setAttribute('fill', "#AAAAAA");
camera.addEventListener('mouseover', function () {
    camera.setAttribute('fill', "#666666");
});
camera.addEventListener('mouseout', function () {
    camera.setAttribute('fill', "#AAAAAA");
});
camera.innerHTML = '<g><path d="M 10.421875 8.773438 C 10.421875 10.09375 9.355469 11.160156 8.039062 11.160156 C 6.71875 11.160156 5.652344 10.09375 5.652344 8.773438 C 5.652344 7.457031 6.71875 6.390625 8.039062 6.390625 C 9.355469 6.390625 10.421875 7.457031 10.421875 8.773438 Z M 10.421875 8.773438"></path><path d="M 14.746094 4.007812 L 12.484375 4.007812 C 12.289062 4.007812 12.117188 3.929688 11.992188 3.785156 C 10.816406 2.457031 10.371094 2.015625 9.882812 2.015625 L 6.320312 2.015625 C 5.828125 2.015625 5.359375 2.460938 4.15625 3.785156 C 4.035156 3.929688 3.835938 4.007812 3.664062 4.007812 L 3.492188 4.007812 L 3.492188 3.664062 C 3.492188 3.492188 3.347656 3.320312 3.148438 3.320312 L 2.066406 3.320312 C 1.894531 3.320312 1.722656 3.464844 1.722656 3.664062 L 1.722656 4.007812 L 1.398438 4.007812 C 0.664062 4.007812 0 4.546875 0 5.285156 L 0 12.609375 C 0 13.347656 0.664062 13.984375 1.398438 13.984375 L 14.722656 13.984375 C 15.460938 13.984375 16 13.320312 16 12.609375 L 16 5.257812 C 16.023438 4.546875 15.484375 4.007812 14.746094 4.007812 Z M 8.210938 12.335938 C 6.121094 12.433594 4.398438 10.714844 4.496094 8.625 C 4.570312 6.804688 6.042969 5.332031 7.886719 5.234375 C 9.976562 5.136719 11.695312 6.855469 11.601562 8.945312 C 11.503906 10.765625 10.027344 12.242188 8.210938 12.335938 Z M 12.042969 6.414062 C 11.75 6.414062 11.503906 6.167969 11.503906 5.875 C 11.503906 5.582031 11.75 5.335938 12.042969 5.335938 C 12.335938 5.335938 12.585938 5.582031 12.585938 5.875 C 12.585938 6.167969 12.335938 6.414062 12.042969 6.414062 Z M 12.042969 6.414062 "></path></g>';
camera.addEventListener('click', function () {
    saveSnapshot();
});
let camerac = document.createElement("div");
camerac.style.padding = "2px"
camerac.append(camera)

// let palette = document.createElementNS("http://www.w3.org/2000/svg", "svg");
// palette.setAttribute('width', "16");
// palette.setAttribute('height', "16");
// palette.setAttribute('viewBox', "0 0 16 16");
// palette.setAttribute('fill', "#AAAAAA");
// palette.addEventListener('mouseover', function () {
//     palette.setAttribute('fill', "#666666");
// });
// palette.addEventListener('mouseout', function () {
//     palette.setAttribute('fill', "#AAAAAA");
// });
// palette.innerHTML = '<g><path d="M 7.984375 0.015625 C 3.601562 0.015625 0 3.617188 0 8 C 0 12.382812 3.601562 15.984375 7.984375 15.984375 C 8.742188 15.984375 9.320312 15.402344 9.320312 14.648438 C 9.320312 14.300781 9.175781 13.980469 8.96875 13.75 C 8.738281 13.519531 8.617188 13.226562 8.617188 12.851562 C 8.617188 12.09375 9.199219 11.515625 9.953125 11.515625 L 11.550781 11.515625 C 13.992188 11.515625 15.992188 9.511719 15.992188 7.070312 C 15.972656 3.210938 12.367188 0.015625 7.984375 0.015625 Z M 3.105469 8 C 2.351562 8 1.773438 7.417969 1.773438 6.664062 C 1.773438 5.914062 2.351562 5.332031 3.105469 5.332031 C 3.859375 5.332031 4.441406 5.914062 4.441406 6.664062 C 4.441406 7.417969 3.863281 8 3.105469 8 Z M 5.777344 4.457031 C 5.023438 4.457031 4.445312 3.875 4.445312 3.121094 C 4.445312 2.367188 5.023438 1.789062 5.777344 1.789062 C 6.53125 1.789062 7.113281 2.367188 7.113281 3.121094 C 7.085938 3.878906 6.535156 4.457031 5.777344 4.457031 Z M 10.195312 4.457031 C 9.4375 4.457031 8.859375 3.875 8.859375 3.121094 C 8.859375 2.367188 9.441406 1.789062 10.195312 1.789062 C 10.945312 1.789062 11.527344 2.367188 11.527344 3.121094 C 11.527344 3.878906 10.945312 4.457031 10.195312 4.457031 Z M 12.863281 8 C 12.105469 8 11.527344 7.417969 11.527344 6.664062 C 11.527344 5.914062 12.109375 5.332031 12.863281 5.332031 C 13.617188 5.332031 14.195312 5.914062 14.195312 6.664062 C 14.195312 7.417969 13.617188 8 12.863281 8 Z M 12.863281 8"/></g>';
// palette.addEventListener('click', function () {
//     switchColorSchema();
// //     infoc.innerHTML = 'Colors: <BR>' + current_schema;
// });
// let palettec = document.createElement("div");
// palettec.style.padding = "2px"
// palettec.append(palette)

let front = document.createElementNS("http://www.w3.org/2000/svg", "svg");
front.setAttribute('width', "16");
front.setAttribute('height', "16");
front.setAttribute('viewBox', "0 0 16 16");
front.setAttribute('fill', "#AAAAAA");
front.setAttribute('stroke', "#AAAAAA");
front.addEventListener('mouseover', function () {
    front.setAttribute('fill', "#666666");
    front.setAttribute('stroke', "#666666");
});
front.addEventListener('mouseout', function () {
    front.setAttribute('fill', "#AAAAAA");
    front.setAttribute('stroke', "#AAAAAA");
});
front.innerHTML = '<g><path d="M 0 5 h 10 v 10 h -10 v -10" fill="none"/><path d="M 4 0 h 10 l -4 4 h -10 l -4 4" stroke="none"/><path d="M 11 5 l 4 -4 v 10 l -4 4 v -10" stroke="none"/></g>';
front.addEventListener('click', function () {
    frontview();
});
let frontc = document.createElement("div");
frontc.style.padding = "2px"
frontc.append(front)

let top = document.createElementNS("http://www.w3.org/2000/svg", "svg");
top.setAttribute('width', "16");
top.setAttribute('height', "16");
top.setAttribute('viewBox', "0 0 16 16");
top.setAttribute('fill', "#AAAAAA");
top.setAttribute('stroke', "#AAAAAA");
top.addEventListener('mouseover', function () {
    top.setAttribute('fill', "#666666");
    top.setAttribute('stroke', "#666666");
});
top.addEventListener('mouseout', function () {
    top.setAttribute('fill', "#AAAAAA");
    top.setAttribute('stroke', "#AAAAAA");
});
top.innerHTML = '<g><path d="M 0 5 h 10 v 10 h -10 v -10" stroke="none"/><path d="M 4 0 h 10 l -4 4 h -10 l -4 4" fill="none"/><path d="M 11 5 l 4 -4 v 10 l -4 4 v -10" stroke="none"/></g>';
top.addEventListener('click', function () {
    topview();
});
let topc = document.createElement("div");
topc.style.padding = "2px"
topc.append(top)

let right = document.createElementNS("http://www.w3.org/2000/svg", "svg");
right.setAttribute('width', "16");
right.setAttribute('height', "16");
right.setAttribute('viewBox', "0 0 16 16");
right.setAttribute('fill', "#AAAAAA");
right.setAttribute('stroke', "#AAAAAA");
right.addEventListener('mouseover', function () {
    right.setAttribute('fill', "#666666");
    right.setAttribute('stroke', "#666666");
});
right.addEventListener('mouseout', function () {
    right.setAttribute('fill', "#AAAAAA");
    right.setAttribute('stroke', "#AAAAAA");
});
right.innerHTML = '<g><path d="M 0 5 h 10 v 10 h -10 v -10" stroke="none"/><path d="M 4 0 h 10 l -4 4 h -10 l -4 4" stroke="none"/><path d="M 11 5 l 4 -4 v 10 l -4 4 v -10" fill="none"/></g>';
right.addEventListener('click', function () {
    rightview();
});
let rightc = document.createElement("div");
rightc.style.padding = "2px"
rightc.append(right)

// let sti = document.createElementNS("http://www.w3.org/2000/svg", "svg");
// sti.setAttribute('width', "16");
// sti.setAttribute('height', "16");
// sti.setAttribute('viewBox', "0 0 16 16");
// sti.setAttribute('fill', "#AAAAAA");
// sti.setAttribute('stroke', "#AAAAAA");
// sti.addEventListener('mouseover', function () {
//     sti.setAttribute('fill', "#666666");
//     sti.setAttribute('stroke', "#666666");
// });
// sti.addEventListener('mouseout', function () {
//     sti.setAttribute('fill', "#AAAAAA");
//     sti.setAttribute('stroke', "#AAAAAA");
// });
// sti.innerHTML = '<g><circle cx="4" cy="4" r="2"/><circle cx="10" cy="2" r="1"/><circle cx="10" cy="12" r="3"/><path d="M 5 5 l 3 4 M 6 3 l 3 -1" ></path></g>';
// sti.addEventListener('click', function () {
//     stickball();
//     updateModel()
// });
// let stic = document.createElement("div");
// stic.style.padding = "2px"
// stic.append(sti)


// let too = document.createElementNS("http://www.w3.org/2000/svg", "svg");
// too.setAttribute('width', "16");
// too.setAttribute('height', "16");
// too.setAttribute('viewBox', "0 0 16 16");
// too.setAttribute('fill', "#AAAAAA");
// too.setAttribute('stroke', "#AAAAAA");
// too.addEventListener('mouseover', function () {
//     too.setAttribute('fill', "#666666");
//     too.setAttribute('stroke', "#666666");
// });
// too.addEventListener('mouseout', function () {
//     too.setAttribute('fill', "#AAAAAA");
//     too.setAttribute('stroke', "#AAAAAA");
// });
// too.innerHTML = '<g><circle cx="4" cy="4" r="1"/><circle cx="10" cy="2" r="1"/><circle cx="10" cy="12" r="1"/><path d="M 4 5 l 5 6 M 5 3 l 4 -1" ></path></g>';
// too.addEventListener('click', function () {
//     toon();
//     updateModel()
// });
// let tooc = document.createElement("div");
// tooc.style.padding = "2px"
// tooc.append(too)

let lic = document.createElementNS("http://www.w3.org/2000/svg", "svg");
lic.setAttribute('width', "16");
lic.setAttribute('height', "16");
lic.setAttribute('viewBox', "0 0 16 16");
lic.setAttribute('fill', "#AAAAAA");
lic.setAttribute('stroke', "#AAAAAA");
lic.addEventListener('mouseover', function () {
    lic.setAttribute('fill', "#666666");
    lic.setAttribute('stroke', "#666666");
});
lic.addEventListener('mouseout', function () {
    lic.setAttribute('fill', "#AAAAAA");
    lic.setAttribute('stroke', "#AAAAAA");
});
lic.innerHTML = '<g><circle cx="4" cy="4" r="2" fill="none"/><circle cx="10" cy="2" r="1" fill="none"/><circle cx="10" cy="12" r="3" fill="none"/><path d="M 5 5 l 3 4 M 6 3 l 3 -1" ></path></g>';
lic.addEventListener('click', function () {
    licorice()
    updateModel();
});
let licc = document.createElement("div");
licc.style.padding = "2px"
licc.append(lic)

let fil = document.createElementNS("http://www.w3.org/2000/svg", "svg");
fil.setAttribute('width', "16");
fil.setAttribute('height', "16");
fil.setAttribute('viewBox', "0 0 16 16");
fil.setAttribute('fill', "#AAAAAA");
fil.setAttribute('stroke', "#AAAAAA");
fil.addEventListener('mouseover', function () {
    fil.setAttribute('fill', "#666666");
    fil.setAttribute('stroke', "#666666");
});
fil.addEventListener('mouseout', function () {
    fil.setAttribute('fill', "#AAAAAA");
    fil.setAttribute('stroke', "#AAAAAA");
});
fil.innerHTML = '<g><circle cx="4" cy="4" r="2"/><circle cx="10" cy="2" r="1"/><circle cx="10" cy="12" r="3"/></g>';
fil.addEventListener('click', function () {
    fill();
    updateModel();
});
let filc = document.createElement("div");
filc.style.padding = "2px"
filc.append(fil)


container.append(canvas)

// topbar.append(infoc)

topbar.append(licc)
// topbar.append(stic)
// topbar.append(tooc)
topbar.append(filc)

topbar.append(frontc)
topbar.append(topc)
topbar.append(rightc)

// topbar.append(palettec)
topbar.append(camerac)

topbar.append(autoscalec)

document.body.appendChild(topbar);
document.body.appendChild(container);

document.body.style.width = "100%"
document.body.style.height = "100%"

speckInteractions({
    container: container,
    scrollZoom: true,
    getRotation: function () { return view.rotation },
    setRotation: function (t: any) { view.rotation = t },
    getTranslation: function () { return view.translation },
    setTranslation: function (t: any) { view.translation = t },
    getZoom: function () { return view.zoom },
    setZoom: function (t: any) { view.zoom = t },
    refreshView: function () { needReset = true; }
});


let saveSnapshot = function () {
    renderer.render(view);
    var imgURL = canvas.toDataURL("image/png");
    var a = document.createElement('a');
    a.href = imgURL;
    a.download = "speck.png";
    document.body.appendChild(a);
    a.click();
}

let setAtomsColor = function (atoms: any) {
    for (const atom in atoms) {
        if (atom in view.elements) {
            view.elements[atom].color = atoms[atom];
            needReset = true;
        }
    }
    if (needReset) {
        speckSystem.calculateBonds(system, view);
        renderer.setSystem(system, view);
    }
}

let setColorSchema = function (schema: string) {
    if (schema in speckColors) {
        current_schema = schema;
        setAtomsColor(speckColors[schema]);
    }
}

let switchColorSchema = function () {
    let update_color = false;
    let first_color = "";
    for (let color in speckColors) {
        if (first_color === "")
            first_color = color;
        if (update_color) {
            setColorSchema(color);
            return;
        }
        if (color === current_schema) {
            update_color = true;
        }
    }
    setColorSchema(first_color);
}

let stickball = function () {
    needReset = true;
    view.atomScale = 0.24;
    view.relativeAtomScale = 0.64;
    view.bondScale = 0.5;
    view.bonds = true;
    view.bondThreshold = 1.2;
    view.brightness = 0.5;
    view.outline = 0.0;
    view.spf = 32;
    view.bondShade = 0.5;
    view.atomShade = 0.5;
    view.dofStrength = 0.0;
    view.dofPosition = 0.5;
    view.ao = 0.75;
    view.spf = 32;
    view.outline = 0;
}

// let toon = function () {
//     stickball()
//     view.atomScale = 0.1;
//     view.relativeAtomScale = 0;
//     view.bondScale = 1;
//     view.bonds = true;
//     view.bondThreshold = 1.2;
// }

let fill = function () {
    stickball()
    view.atomScale = 0.6;
    view.relativeAtomScale = 1.0;
    view.bonds = false;
}

let licorice = function () {
    stickball()
    view.ao = 0;
    view.spf = 0;
    view.outline = 1;
    view.bonds = true;
}

let updateModel = function () {
    Streamlit.setComponentValue({
        'bonds': view.bonds,
        'atomScale': view.atomScale,
        'relativeAtomScale': view.relativeAtomScale,
        'bondScale': view.bondScale,
        'brightness': view.brightness,
        'outline': view.outline,
        'spf': view.spf,
        'bondShade': view.bondShade,
        'atomShade': view.atomShade,
        'dofStrength': view.dofStrength,
        'dofPosition': view.dofPosition,
        'ao': view.ao,
        'aoRes': view.aoRes
    })
}

let loadStructure = function (tdata: string) {
    system = undefined;
    var data = xyz(tdata)[0];
    if (data) {
        system = speckSystem.new();
        for (var i = 0; i < data.length; i++) {
            var a = data[i];
            var x = a.position[0];
            var y = a.position[1];
            var z = a.position[2];
            speckSystem.addAtom(system, a.symbol, x, y, z);
        }
        center();
    }
}


let center = function () {
    if (system) {
        speckSystem.center(system);
        speckSystem.calculateBonds(system, view);
        renderer.setSystem(system, view);
        speckView.center(view, system);
        needReset = true;
    }
}

let topview = function () {
    if (system) {
        speckView.rotateX(view, Math.PI / 2);
        center();
    }
}

let frontview = function () {
    if (system) {
        speckView.rotateX(view, 0);
        center();
    }
}

let rightview = function () {
    if (system) {
        speckView.rotateY(view, -Math.PI / 2);
        center();
    }
}

let xyz = function (data: string) {
    var lines = data.split('\n');
    var natoms = parseInt(lines[0]);
    var nframes = Math.floor(lines.length / (natoms + 2));
    var trajectory = []
    for (var i = 0; i < nframes; i++) {
        var atoms = [];
        type ATOM = {
            [key: string]: any;
        };
        for (var j = 0; j < natoms; j++) {
            var line = lines[i * (natoms + 2) + j + 2].split(/\s+/);
            var atom: ATOM = {};
            var k = 0;
            while (line[k] === "") k++;
            atom.symbol = line[k++];
            atom.position = [parseFloat(line[k++]), parseFloat(line[k++]), parseFloat(line[k++])];
            atoms.push(atom);
        }
        trajectory.push(atoms);
    }
    return trajectory;
}


let reflow = function () {
    var ww = document.body.clientWidth;
    var wh = document.body.clientHeight;
    if (ww === 0)
        ww = view.resolution.x;
    if (wh === 0)
        wh = view.resolution.y;
    if (view.resolution.x === ww && view.resolution.y === wh)
        return;
    container.style.height = wh + "px";
    container.style.width = ww + "px";
    container.style.left = 0 + "px";
    container.style.top = 0 + "px";
    view.resolution.x = ww;
    view.resolution.y = wh;
    renderer = new speckRenderer(canvas, view.resolution, view.aoRes);
}

let loop = function () {
    if (needReset) {
        renderer.reset();
        needReset = false;
    }
    renderer.render(view);
    requestAnimationFrame(function () { loop() });
}

/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event: Event): void {
    // Get the RenderData from the event
    const data = (event as CustomEvent<RenderData>).detail
    console.log(data.args);
    document.body.style.width = data.args["width"]
    document.body.style.height = data.args["height"]
    view.bonds = data.args['bonds'];
    view.atomScale = data.args['atomScale'];
    view.relativeAtomScale = data.args['relativeAtomScale'];
    view.bondScale = data.args['bondScale'];
    view.brightness = data.args['brightness'];
    view.outline = data.args['outline'];
    view.spf = data.args['spf'];
    view.bondShade = data.args['bondShade'];
    view.atomShade = data.args['atomShade'];
    view.dofStrength = data.args['dofStrength'];
    view.dofPosition = data.args['dofPosition'];
    view.ao = data.args['ao'];
    view.aoRes = data.args['aoRes'];
    reflow()
    loop();
    loadStructure(data.args["data"]);
    // We tell Streamlit to update our frameHeight after each render event, in
    // case it has changed. (This isn't strictly necessary for the example
    // because our height stays fixed, but this is a low-cost function, so
    // there's no harm in doing it redundantly.)
    Streamlit.setFrameHeight()
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady()

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight()

