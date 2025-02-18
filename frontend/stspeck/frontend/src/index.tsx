import { Streamlit, RenderData } from "streamlit-component-lib"

const speckRenderer = require('./renderer.js');
const speckSystem = require('./system.js');
const speckView = require('./view.js');
const speckInteractions = require('./interactions.js');
const speckColors = require('./colors.js');


let system = speckSystem.new();
let view = speckView.new();
view.resolution.x = 512;
view.resolution.y = 512;
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
let current_schema = "speck_1";
var view_position = "front";


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
topbar.style.background = "rgba(0, 0, 0, 0)"
topbar.style.flexDirection = "row";
topbar.style.alignContent = "flex-end";
topbar.style.display = "flex";



let infoc = document.createElement("div");
infoc.style.fontSize = "10px";
infoc.style.color = "#AAA";



let autoscale = document.createElementNS("http://www.w3.org/2000/svg", "svg");

autoscale.setAttribute('width', "16");
autoscale.setAttribute('height', "16");
autoscale.setAttribute('viewBox', "0 0 20 20");
autoscale.setAttribute('fill', "#AAAAAA");
autoscale.setAttribute('stroke', "#AAAAAA");

autoscale.addEventListener('mouseover', function () {
    autoscale.setAttribute('stroke', "#666666");
});
autoscale.addEventListener('mouseout', function () {
    autoscale.setAttribute('stroke', "#AAAAAA");
});
autoscale.innerHTML = '<g fill="none" stroke-width="2" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" transform="translate(2 2)"><path d="m16.5 5.5v-3c0-1.1045695-.8954305-2-2-2h-3"/><path d="m8.5 10.5v-4"/><path d="m6.5 8.5h4"/><path d="m16.5 11.5v3c0 1.1045695-.8954305 2-2 2h-3m-6-16h-3c-1.1045695 0-2 .8954305-2 2v3m5 11h-3c-1.1045695 0-2-.8954305-2-2v-3"/></g>'

autoscale.addEventListener('click', function () {
    center();
});

let autoscalec = document.createElement("div");
autoscalec.style.padding = "2px 0px 0px 0px"
autoscalec.append(autoscale)

let camera = document.createElementNS("http://www.w3.org/2000/svg", "svg");
camera.setAttribute('width', "16");
camera.setAttribute('height', "16");
camera.setAttribute('viewBox', "0 0 22 22");
camera.setAttribute('fill', "#AAAAAA");
camera.addEventListener('mouseover', function () {
    camera.setAttribute('fill', "#666666");
});
camera.addEventListener('mouseout', function () {
    camera.setAttribute('fill', "#AAAAAA");
});
camera.innerHTML = '<g><path fill-rule="evenodd" clip-rule="evenodd" d="M8 4C8 3.44772 8.41328 3 8.92308 3H15.0769C15.5867 3 16 3.44772 16 4C16 4.55228 15.5867 5 15.0769 5H8.92308C8.41328 5 8 4.55228 8 4Z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M9.77778 21H14.2222C17.3433 21 18.9038 21 20.0248 20.2777C20.51 19.965 20.9267 19.5632 21.251 19.0953C22 18.0143 22 16.5095 22 13.4999C22 10.4903 21.9998 8.9857 21.2508 7.90473C20.9266 7.43676 20.5099 7.03497 20.0246 6.72228C18.9036 6 17.3431 6 14.2221 6H9.77761C6.65659 6 5.09607 6 3.97508 6.72228C3.48979 7.03497 3.07312 7.43676 2.74886 7.90473C2 8.98547 2 10.4896 2 13.4979L2 13.4999C2 16.5095 2 18.0143 2.74902 19.0953C3.07328 19.5632 3.48995 19.965 3.97524 20.2777C5.09624 21 6.65675 21 9.77778 21ZM7.83333 13.4999C7.83333 11.2808 9.69881 9.48196 12 9.48196C14.3012 9.48196 16.1667 11.2808 16.1667 13.4999C16.1667 15.7189 14.3012 17.5178 12 17.5178C9.69881 17.5178 7.83333 15.7189 7.83333 13.4999ZM9.5 13.4999C9.5 12.1685 10.6193 11.0891 12 11.0891C13.3807 11.0891 14.5 12.1685 14.5 13.4999C14.5 14.8313 13.3807 15.9106 12 15.9106C10.6193 15.9106 9.5 14.8313 9.5 13.4999ZM18.1111 9.48196C17.6509 9.48196 17.2778 9.84174 17.2778 10.2855C17.2778 10.7294 17.6509 11.0891 18.1111 11.0891H18.6667C19.1269 11.0891 19.5 10.7294 19.5 10.2855C19.5 9.84174 19.1269 9.48196 18.6667 9.48196H18.1111Z"/></g>'

camera.addEventListener('click', function () {
    saveSnapshot();
});
let camerac = document.createElement("div");
camerac.style.padding = "2px 2px 0px 0px"
camerac.append(camera)

let palette = document.createElementNS("http://www.w3.org/2000/svg", "svg");
palette.setAttribute('width', "16");
palette.setAttribute('height', "16");
palette.setAttribute('viewBox', "0 0 16 16");
palette.setAttribute('fill', "#AAAAAA");
palette.addEventListener('mouseover', function () {
    palette.setAttribute('opacity', "0.5");
});
palette.addEventListener('mouseout', function () {
    palette.setAttribute('opacity', "1");
});
palette.innerHTML = '<svg><path d="M10.5 10.5c.002 2.762-2.237 5-5 5s-5.002-2.238-5-5c-.002-2.76 2.237-5 5-5s5.002 2.24 5 5z" color="#000000" fill="#ff15a1" stroke="#373737" stroke-width=".49999682600000006"/><path d="M8 1.401a4.998 4.998 0 0 0-2.488 9.334c-.004-.078-.012-.155-.012-.234a4.998 4.998 0 0 1 7.488-4.334A4.994 4.994 0 0 0 8 1.4z" fill="#1583ff"/><path d="M10.5 5.5a4.998 4.998 0 0 0-5 5c0 .08.008.157.012.235A4.998 4.998 0 0 0 13 6.401c0-.079-.008-.156-.012-.234A4.975 4.975 0 0 0 10.5 5.5z" fill="#00cf2d"/><path d="M12.988 6.167c.004.078.012.155.012.234a4.998 4.998 0 0 1-7.489 4.334 4.994 4.994 0 0 0 4.989 4.766 4.998 4.998 0 0 0 2.488-9.334z" fill="#f8ff15"/><path d="M5.512 10.735a4.996 4.996 0 0 0 2.486 4.093 4.987 4.987 0 0 0 2.49-4.091A4.978 4.978 0 0 1 8 11.4a4.975 4.975 0 0 1-2.488-.666z" fill="#ef0000"/><path d="M7.998 6.173A4.991 4.991 0 0 0 5.5 10.5c0 .079.008.156.012.234a4.978 4.978 0 0 0 4.977.002c.003-.079.011-.157.011-.236a4.99 4.99 0 0 0-2.502-4.328z" fill="#383027"/><path d="M5.5 5.5c-.91 0-1.76.247-2.494.67a4.99 4.99 0 0 0 2.506 4.564c-.004-.077-.012-.154-.012-.233a4.991 4.991 0 0 1 2.498-4.328A4.975 4.975 0 0 0 5.5 5.5z" fill="#5100cc"/><path d="M8 1.401a4.998 4.998 0 0 0-4.994 4.77 4.998 4.998 0 1 0 4.992 8.658 4.998 4.998 0 1 0 4.99-8.662A4.994 4.994 0 0 0 8 1.4z" fill="none" stroke="#373737" stroke-width=".9999936520000001"/></svg>'

palette.addEventListener('click', function () {
    switchColorSchema();
});
let palettec = document.createElement("div");
palettec.style.padding = "2px"
palettec.append(palette)

let front = document.createElementNS("http://www.w3.org/2000/svg", "svg");
front.setAttribute('width', "20");
front.setAttribute('height', "20");
front.setAttribute('viewBox', "0 0 24 24");
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
front.innerHTML = '<svg stroke-width = "0.5"><path d="M11.2797426,15.9868494 L10.1464466,14.8535534 C9.95118446,14.6582912 9.95118446,14.3417088 10.1464466,14.1464466 C10.3417088,13.9511845 10.6582912,13.9511845 10.8535534,14.1464466 L12.8535534,16.1464466 C13.0488155,16.3417088 13.0488155,16.6582912 12.8535534,16.8535534 L10.8535534,18.8535534 C10.6582912,19.0488155 10.3417088,19.0488155 10.1464466,18.8535534 C9.95118446,18.6582912 9.95118446,18.3417088 10.1464466,18.1464466 L11.3044061,16.9884871 C10.3667147,16.9573314 9.46306739,16.8635462 8.61196501,16.7145167 C9.33747501,19.2936084 10.6229353,21 12,21 C14.0051086,21 15.8160018,17.3821896 15.9868494,12.7202574 L14.8535534,13.8535534 C14.6582912,14.0488155 14.3417088,14.0488155 14.1464466,13.8535534 C13.9511845,13.6582912 13.9511845,13.3417088 14.1464466,13.1464466 L16.1464466,11.1464466 C16.3417088,10.9511845 16.6582912,10.9511845 16.8535534,11.1464466 L18.8535534,13.1464466 C19.0488155,13.3417088 19.0488155,13.6582912 18.8535534,13.8535534 C18.6582912,14.0488155 18.3417088,14.0488155 18.1464466,13.8535534 L16.9884871,12.6955939 C16.8167229,17.8651676 14.7413901,22 12,22 C9.97580598,22 8.3147521,19.7456544 7.515026,16.484974 C4.2543456,15.6852479 2,14.024194 2,12 C2,9.97580598 4.2543456,8.3147521 7.515026,7.515026 C8.3147521,4.2543456 9.97580598,2 12,2 C13.5021775,2 14.8263891,3.23888365 15.7433738,5.30744582 C15.8552836,5.55989543 15.7413536,5.8552671 15.4889039,5.96717692 C15.2364543,6.07908673 14.9410827,5.96515672 14.8291729,5.71270711 C14.0550111,3.96632921 13.0221261,3 12,3 C10.6229353,3 9.33747501,4.70639159 8.61196501,7.28548333 C9.67174589,7.09991387 10.812997,7 12,7 C17.4892085,7 22,9.13669069 22,12 C22,13.5021775 20.7611164,14.8263891 18.6925542,15.7433738 C18.4401046,15.8552836 18.1447329,15.7413536 18.0328231,15.4889039 C17.9209133,15.2364543 18.0348433,14.9410827 18.2872929,14.8291729 C20.0336708,14.0550111 21,13.0221261 21,12 C21,9.89274656 17.0042017,8 12,8 C10.6991081,8 9.46636321,8.12791023 8.35424759,8.35424759 C8.12791023,9.46636321 8,10.6991081 8,12 C8,13.3008919 8.12791023,14.5336368 8.35424759,15.6457524 C9.25899447,15.8298862 10.2435788,15.9488767 11.2797426,15.9868494 Z M7.28548333,8.61196501 C4.70639159,9.33747501 3,10.6229353 3,12 C3,13.3770647 4.70639159,14.662525 7.28548333,15.388035 C7.09991387,14.3282541 7,13.187003 7,12 C7,10.812997 7.09991387,9.67174589 7.28548333,8.61196501 L7.28548333,8.61196501 Z"/></svg>'
front.addEventListener('click', function () {
      rotate();
});
let frontc = document.createElement("div");
frontc.style.padding = "2px"
frontc.append(front)


let sti = document.createElementNS("http://www.w3.org/2000/svg", "svg");
sti.setAttribute('width', "16");
sti.setAttribute('height', "16");
sti.setAttribute('viewBox', "0 0 16 16");
sti.setAttribute('fill', "#AAAAAA");
sti.setAttribute('stroke', "#AAAAAA");
sti.addEventListener('mouseover', function () {
    sti.setAttribute('fill', "#666666");
    sti.setAttribute('stroke', "#666666");
});
sti.addEventListener('mouseout', function () {
    sti.setAttribute('fill', "#AAAAAA");
    sti.setAttribute('stroke', "#AAAAAA");
});
sti.innerHTML = '<g><circle cx="4" cy="4" r="2"/><circle cx="10" cy="2" r="1"/><circle cx="10" cy="12" r="3"/><path d="M 5 5 l 3 4 M 6 3 l 3 -1" ></path></g>';
sti.addEventListener('click', function () {
    stickball();
    updateModel()
});
let stic = document.createElement("div");
stic.style.padding = "2px"
stic.append(sti)


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


topbar.append(stic)
topbar.append(licc)
topbar.append(filc)

topbar.append(frontc)

topbar.append(palettec)
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
    a.download = "molecule.png";
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
        speckSystem.updateBondsColor(system, view);
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
    view.relativeAtomScale = 1;
    view.bondScale = 0.72;
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


let fill = function () {
    stickball()
    view.atomScale = 0.52;
    view.relativeAtomScale = 1.0;
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

// Our function to load bonds directly to the view from json
let loadStructureFromJson = function (json_data: any) {
    system = undefined;
    system = speckSystem.new();
    var atoms = json_data.atoms
    var bonds = json_data.bonds
    for (var i = 0; i < atoms.length; i++) {
            speckSystem.addAtom(system, atoms[i].symbol, atoms[i].x, atoms[i].y, atoms[i].z);
        }
        center();
    for (var j = 0; j < bonds.length; j++) {
            var idxA = bonds[j].begin_atom;
            var idxB = bonds[j].end_atom;
            speckSystem.addBond(system, idxA, idxB);
        }
        center();
   }


let center = function () {
    if (system) {
        speckSystem.center(system);
        renderer.setSystem(system, view);
        speckView.center(view, system);
        needReset = true;
    }
}


let rotate = function () {
      if (system) {
        if (view_position === 'front') {
        speckView.rotateX(view, Math.PI / 2);
        center();
        view_position = 'top';
        return
        };
        if (view_position === 'top') {
        speckView.rotateY(view, -Math.PI / 2);
        center();
        view_position = 'right';
        return
        };
        if (view_position === 'right') {
        speckView.rotateX(view, 0);
        center();
        view_position = 'front';
        return
        }
    }
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
    loadStructureFromJson(data.args["data"]);
    setColorSchema(current_schema);

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

