"use strict";

var glm = require("./gl-matrix")

var elements = require("./elements");
var consts = require("./const");

var newSystem = module.exports.new = function() {
    return {
        atoms: [],
        farAtom: undefined,
        bonds: []
    }
};

var addAtom = module.exports.addAtom = function(s, symbol, x, y, z) {
    s.atoms.push({
        symbol: symbol,
        x: x,
        y: y,
        z: z,
    });
};

// Our function to Add bonds explicitly [Not Finished]
var addBond = module.exports.addBond = function(s, idxA, idxB) {
        var elems = elements;
        var atom_a = s.atoms[idxA]
        var atom_b = s.atoms[idxB]
        var ea = elems[atom_a.symbol]
        var eb = elems[atom_b.symbol]

        var l = glm.vec3.fromValues(atom_a.x, atom_a.y, atom_a.z);
        var m = glm.vec3.fromValues(atom_b.x, atom_b.y, atom_b.z);
        var d = glm.vec3.distance(l, m);

        s.bonds.push({
                        begin : idxA,
                        end: idxB,
                        posA: {
                            x: atom_a.x,
                            y: atom_a.y,
                            z: atom_a.z
                        },
                        posB: {
                            x: atom_b.x,
                            y: atom_b.y,
                            z: atom_b.z
                        },
                        radA: ea.radius,
                        radB: eb.radius,
                        colA: {
                            r: ea.color[0],
                            g: ea.color[1],
                            b: ea.color[2]
                        },
                        colB: {
                            r: eb.color[0],
                            g: eb.color[1],
                            b: eb.color[2]
                        },
                       // cutoff: d/(ea.radius+eb.radius)
                       cutoff: 0
                    });
};

var updateBondsColor = module.exports.updateBondsColor = function(s, v) {
    var elems = elements;
    if (v != undefined)
       elems = v.elements;
    for (var i = 0; i < s.bonds.length; i++) {
        var atom_a = s.atoms[s.bonds[i].begin]
        var atom_b = s.atoms[s.bonds[i].end]
        var ea = elems[atom_a.symbol];
        var eb = elems[atom_b.symbol];
        s.bonds[i].colA = {
                            r: ea.color[0],
                            g: ea.color[1],
                            b: ea.color[2]
                        }
        s.bonds[i].colB = {
                            r: eb.color[0],
                            g: eb.color[1],
                            b: eb.color[2]
                        }

    };
}

//var calculateBonds = module.exports.calculateBonds = function(s, v) {
//    var elems = elements;
//    if (v != undefined)
//        elems = v.elements;
//    var bonds = [];
//    var sorted = s.atoms.slice();
//    sorted.sort(function(a, b) {
//        return a.z - b.z;
//    });
//    for (var i = 0; i < sorted.length; i++) {
//        var a = sorted[i];
//        var j = i + 1;
//        while(j < sorted.length && sorted[j].z < sorted[i].z + 2.5 * 2 * consts.MAX_ATOM_RADIUS) {
//            var b = sorted[j];
//            var l = glm.vec3.fromValues(a.x, a.y, a.z);
//            var m = glm.vec3.fromValues(b.x, b.y, b.z);
//            var d = glm.vec3.distance(l, m);
//            var ea = elems[a.symbol];
//            var eb = elems[b.symbol];
//            if (d < 2.5*(ea.radius+eb.radius)) {
//                bonds.push({
//                    posA: {
//                        x: a.x,
//                        y: a.y,
//                        z: a.z
//                    },
//                    posB: {
//                        x: b.x,
//                        y: b.y,
//                        z: b.z
//                    },
//                    radA: ea.radius,
//                    radB: eb.radius,
//                    colA: {
//                        r: ea.color[0],
//                        g: ea.color[1],
//                        b: ea.color[2]
//                    },
//                    colB: {
//                        r: eb.color[0],
//                        g: eb.color[1],
//                        b: eb.color[2]
//                    },
//                    cutoff: d/(ea.radius+eb.radius)
//                });
//            }
//            j++;
//        }
//    }
//    bonds.sort(function(a, b) {
//        return a.cutoff - b.cutoff;
//    });
//    s.bonds = bonds;
//}

var getCentroid = module.exports.getCentroid = function(s) {
    var xsum = 0;
    var ysum = 0;
    var zsum = 0;
    for (var i = 0; i < s.atoms.length; i++) {
        xsum += s.atoms[i].x;
        ysum += s.atoms[i].y;
        zsum += s.atoms[i].z;
    }
    return {
        x: xsum/s.atoms.length,
        y: ysum/s.atoms.length,
        z: zsum/s.atoms.length
    };
};

var center = module.exports.center = function(s) {
    var shift = getCentroid(s);
    for (var i = 0; i < s.atoms.length; i++) {
        var atom = s.atoms[i];
        atom.x -= shift.x;
        atom.y -= shift.y;
        atom.z -= shift.z;
    }
}

var getFarAtom = module.exports.getFarAtom = function(s, v) {
    if (s.farAtom !== undefined) {
        return s.farAtom;
    }
    var elems = elements;
    if (v != undefined)
        elems = v.elements;

    s.farAtom = s.atoms[0];
    var maxd = 0.0;
    for (var i = 0; i < s.atoms.length; i++) {
        var atom = s.atoms[i];
        var r = elems[atom.symbol].radius;
        var rd = Math.sqrt(r*r + r*r + r*r) * 2.5;
        var d = Math.sqrt(atom.x*atom.x + atom.y*atom.y + atom.z*atom.z) + rd;
        if (d > maxd) {
            maxd = d;
            s.farAtom = atom;
        }
    }
    return s.farAtom;
}

var getRadius = module.exports.getRadius = function(s, v) {
    var atom = getFarAtom(s, v);
    var r = consts.MAX_ATOM_RADIUS;
    var rd = Math.sqrt(r*r + r*r + r*r) * 2.5;
    return Math.sqrt(atom.x*atom.x + atom.y*atom.y + atom.z*atom.z) + rd;
}
