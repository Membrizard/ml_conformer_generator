module.exports = {};

var reference_struct_colors = [
                               [1.000, 0.000, 1.000],
                               [1.000, 1.000, 1.000],
                               [0.596, 0.921, 0.204],
                               ]

var standard_schema = {
  'Ac' : [0.439, 0.671, 0.980],
  'Ag' : [0.753, 0.753, 0.753],
  'Al' : [0.749, 0.651, 0.651],
  'Am' : [0.329, 0.361, 0.949],
  'Ar' : [0.502, 0.820, 0.890],
  'As' : [0.741, 0.502, 0.890],
  'At' : [0.459, 0.310, 0.271],
  'Au' : [1.000, 0.820, 0.137],
  'B' : [1.000, 0.710, 0.710],
  'Ba' : [0.000, 0.788, 0.000],
  'Be' : [0.761, 1.000, 0.000],
  'Bh' : [0.878, 0.000, 0.220],
  'Bi' : [0.620, 0.310, 0.710],
  'Bk' : [0.541, 0.310, 0.890],
  'Br' : [0.651, 0.161, 0.161],
  'C' : [0.565, 0.565, 0.565],
  'Ca' : [0.239, 1.000, 0.000],
  'Cd' : [1.000, 0.851, 0.561],
  'Ce' : [1.000, 1.000, 0.780],
  'Cf' : [0.631, 0.212, 0.831],
  'Cl' : [0.122, 0.941, 0.122],
  'Cm' : [0.471, 0.361, 0.890],
  'Co' : [0.941, 0.565, 0.627],
  'Cr' : [0.541, 0.600, 0.780],
  'Cs' : [0.341, 0.090, 0.561],
  'Cu' : [0.784, 0.502, 0.200],
  'Db' : [0.820, 0.000, 0.310],
  'Dy' : [0.122, 1.000, 0.780],
  'Er' : [0.000, 0.902, 0.459],
  'Es' : [0.702, 0.122, 0.831],
  'Eu' : [0.380, 1.000, 0.780],
  'F' : [0.565, 0.878, 0.314],
  'Fe' : [0.878, 0.400, 0.200],
  'Fm' : [0.702, 0.122, 0.729],
  'Fr' : [0.259, 0.000, 0.400],
  'Ga' : [0.761, 0.561, 0.561],
  'Gd' : [0.271, 1.000, 0.780],
  'Ge' : [0.400, 0.561, 0.561],
  'H' : [1.000, 1.000, 1.000],
  'He' : [0.851, 1.000, 1.000],
  'Hf' : [0.302, 0.761, 1.000],
  'Hg' : [0.722, 0.722, 0.816],
  'Ho' : [0.000, 1.000, 0.612],
  'Hs' : [0.902, 0.000, 0.180],
  'I' : [0.580, 0.000, 0.580],
  'In' : [0.651, 0.459, 0.451],
  'Ir' : [0.090, 0.329, 0.529],
  'K' : [0.561, 0.251, 0.831],
  'Kr' : [0.361, 0.722, 0.820],
  'La' : [0.439, 0.831, 1.000],
  'Li' : [0.800, 0.502, 1.000],
  'Lr' : [0.780, 0.000, 0.400],
  'Lu' : [0.000, 0.671, 0.141],
  'Md' : [0.702, 0.051, 0.651],
  'Mg' : [0.541, 1.000, 0.000],
  'Mn' : [0.611, 0.478, 0.780],
  'Mo' : [0.329, 0.710, 0.710],
  'Mt' : [0.922, 0.000, 0.149],
  'N' : [0.188, 0.314, 0.973],
  'Na' : [0.671, 0.361, 0.949],
  'Nb' : [0.451, 0.761, 0.788],
  'Nd' : [0.780, 1.000, 0.780],
  'Ne' : [0.702, 0.890, 0.961],
  'Ni' : [0.314, 0.816, 0.314],
  'No' : [0.741, 0.051, 0.529],
  'Np' : [0.000, 0.502, 1.000],
  'O' : [1.000, 0.051, 0.051],
  'Os' : [0.149, 0.400, 0.588],
  'P' : [1.000, 0.502, 0.000],
  'Pa' : [0.000, 0.631, 1.000],
  'Pb' : [0.341, 0.349, 0.380],
  'Pd' : [0.000, 0.412, 0.522],
  'Pm' : [0.639, 1.000, 0.780],
  'Po' : [0.671, 0.361, 0.000],
  'Pr' : [0.851, 1.000, 0.780],
  'Pt' : [0.816, 0.816, 0.878],
  'Pu' : [0.000, 0.420, 1.000],
  'Ra' : [0.000, 0.490, 0.000],
  'Rb' : [0.439, 0.180, 0.690],
  'Re' : [0.149, 0.490, 0.671],
  'Rf' : [0.800, 0.000, 0.349],
  'Rh' : [0.039, 0.490, 0.549],
  'Rn' : [0.259, 0.510, 0.588],
  'Ru' : [0.141, 0.561, 0.561],
  'S' : [1.000, 1.000, 0.188],
  'Sb' : [0.620, 0.388, 0.710],
  'Sc' : [0.902, 0.902, 0.902],
  'Se' : [1.000, 0.631, 0.000],
  'Sg' : [0.851, 0.000, 0.271],
  'Si' : [0.941, 0.784, 0.627],
  'Sm' : [0.561, 1.000, 0.780],
  'Sn' : [0.400, 0.502, 0.502],
  'Sr' : [0.000, 1.000, 0.000],
  'Ta' : [0.302, 0.651, 1.000],
  'Tb' : [0.189, 1.000, 0.780],
  'Tc' : [0.231, 0.620, 0.620],
  'Te' : [0.831, 0.478, 0.000],
  'Th' : [0.000, 0.729, 1.000],
  'Ti' : [0.749, 0.761, 0.780],
  'Tl' : [0.651, 0.329, 0.302],
  'Tm' : [0.000, 0.831, 0.322],
  'U' : [0.000, 0.561, 1.000],
  'V' : [0.651, 0.651, 0.671],
  'W' : [0.129, 0.580, 0.839],
  'Xe' : [0.259, 0.620, 0.690],
  'Y' : [0.580, 1.000, 1.000],
  'Yb' : [0.000, 0.749, 0.220],
  'Zn' : [0.490, 0.502, 0.690],
  'Zr' : [0.580, 0.878, 0.878],
}

for (var i = 0; i < reference_struct_colors.length; i++) {
    var name = 'speck_' + i
    var schema = Object.assign({}, standard_schema)
    schema['Ref'] = reference_struct_colors[i]
    module.exports[name] = schema
}
