"""Example showing how to display a vega contour plot."""
from absl import app
import numpy as np

import multiscope
from multiscope.examples import examples

_CONTOUR_SPEC = {
    '$schema':
        'https://vega.github.io/schema/vega/v5.json',
    'description':
        'A contour plot example that overlays multiple contours '
        '(some derived) and some markers.',
    'width':
        600,
    'height':
        600,
    'scales': [{
        'name': 'color',
        'type': 'linear',
        'domain': [0, 1],
        'range': {
            'scheme': 'plasma'
        }
    }],
    'marks': [{
        'type':
            'group',
        'marks': [{
            'type': 'path',
            'from': {
                'data': 'contours'
            },
            'encode': {
                'enter': {
                    'stroke': {
                        'value': 'black'
                    },
                    'strokeWidth': {
                        'value': 0.5
                    },
                    'fill': {
                        'scale': 'color',
                        'field': 'contour.value'
                    }
                }
            },
            'transform': [{
                'type': 'geopath',
                'field': 'datum.contour'
            }]
        }, {
            'type': 'path',
            'from': {
                'data': 'middle_contour'
            },
            'encode': {
                'enter': {
                    'stroke': {
                        'value': 'black'
                    },
                    'strokeWidth': {
                        'value': 2
                    },
                }
            },
            'transform': [{
                'type': 'geopath',
                'field': 'datum.contour'
            }]
        }],
    }, {
        'type':
            'group',
        'scales': [{
            'name': 'x',
            'type': 'linear',
            'range': 'width',
            'zero': False,
            'domain': [-1, 1]
        }, {
            'name': 'y',
            'type': 'linear',
            'range': 'height',
            'zero': False,
            'domain': [-1, 1]
        }],
        'axes': [{
            'orient': 'bottom',
            'scale': 'x',
            'tickCount': 5,
        }, {
            'orient': 'right',
            'scale': 'y',
            'tickCount': 5,
        }],
        'marks': [{
            'name': 'center',
            'type': 'symbol',
            'from': {
                'data': 'center'
            },
            'encode': {
                'update': {
                    'xc': {
                        'scale': 'x',
                        'field': 'x'
                    },
                    'yc': {
                        'scale': 'y',
                        'field': 'y'
                    },
                    'shape': {
                        'value': 'M0,-1V1M-1,0H1'
                    },
                    'strokeWidth': {
                        'value': 2
                    },
                    'stroke': {
                        'value': 'magenta'
                    },
                    'fill': {
                        'value': 'transparent'
                    }
                }
            }
        }]
    }]
}


def generate_flux(x0, y0, sigma_x, sigma_y, resolution):
  x = np.linspace(-1, 1, resolution, True) - x0
  y = np.linspace(-1, 1, resolution) - y0

  gx = np.exp(-x**2 / (2 * sigma_x**2))
  gy = np.exp(-y**2 / (2 * sigma_y**2))
  g = np.outer(gx, gy)
  return g


def get_data_spec(x0, y0, sigma_x, sigma_y, resolution):
  flux = generate_flux(x0, y0, sigma_x, sigma_y, resolution)
  thresholds = sorted(np.linspace(flux.min(), flux.max(), 20).tolist())
  return [{
      'name': 'flux',
      'values': {
          'width': flux.shape[0],
          'height': flux.shape[1],
          'values': flux.flatten().tolist(),
      }
  }, {
      'name': 'center',
      'values': [{
          'x': y0,
          'y': -x0
      }]
  }, {
      'name':
          'contours',
      'source':
          'flux',
      'transform': [{
          'type': 'isocontour',
          'scale': {
              'expr': 'width / datum.width'
          },
          'smooth': True,
          'thresholds': thresholds
      }]
  }, {
      'name':
          'middle_contour',
      'source':
          'flux',
      'transform': [{
          'type': 'isocontour',
          'scale': {
              'expr': 'width / datum.width'
          },
          'smooth': True,
          'thresholds': [thresholds[int(len(thresholds) / 2)]]
      }]
  }]


def perturbation():
  return np.random.uniform(-.0025, .0025)


def main(_):
  multiscope.start_server()
  w = multiscope.DataSpecWriter('Dynamic Contours')
  w.set_spec(_CONTOUR_SPEC)

  resolution = 30
  x0 = np.random.uniform(-0.25, 0.25)
  y0 = np.random.uniform(-0.25, 0.25)
  sigma_x = np.random.uniform(0.25, 0.5)
  sigma_y = np.random.uniform(0.25, 0.5)
  for _ in examples.step():
    x0 += perturbation()
    y0 += perturbation()
    sigma_x += perturbation()
    sigma_y += perturbation()
    w.write(get_data_spec(x0, y0, sigma_x, sigma_y, resolution))


if __name__ == '__main__':
  app.run(main)
