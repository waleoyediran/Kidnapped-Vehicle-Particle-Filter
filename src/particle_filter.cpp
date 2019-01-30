/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  num_particles = 1000;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++) {
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen); 

    Particle p = {
      i, // id
      sample_x, // x
      sample_y, // y
      sample_theta, // theta
      1.0 // weight
    };

    particles.push_back(p);

    weights.push_back(1.0); // TODO: Check?
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  if (yaw_rate == 0.0) {
    yaw_rate = 0.0000000000000000001;
  }

  for (int i=0; i<num_particles; i++) {
    Particle p = particles[i];

    double p_x = p.x + (velocity/yaw_rate) 
      * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
    double p_y = p.y + (velocity/yaw_rate) 
      * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
    double p_theta = p.theta + (yaw_rate * delta_t);

    normal_distribution<double> dist_x(p_x, std_pos[0]);
    normal_distribution<double> dist_y(p_y, std_pos[1]);
    normal_distribution<double> dist_theta(p_theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    particles[i] = p; // TODO: Is this necessary?
  } 

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (int i=0; i<observations.size(); i++) {
    int closest_id = 0;
    double closest_distance = std::numeric_limits<double>::max();

    for (int j=0; j<predicted.size(); j++) {
      double current_distance = dist(
        predicted[j].x, predicted[j].y, 
        observations[i].x, observations[i].y
      );

      if (current_distance < closest_distance) {
        closest_id = j;
        closest_distance = current_distance;
        // TODO: Implement a dict of previously associated predictions
      }
    }

    observations[i].id = closest_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i=0; i<particles.size(); i++) {
    Particle p = particles[i];

    // transform observation to map coordinate
    vector<LandmarkObs> observations_mc;
    for (int j=0; j<observations.size(); j++) {
      LandmarkObs obs = observations[j];

      double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      double y_map = p.y + (sin(p.theta) * obs.x) - (cos(p.theta) * obs.y);

      observations_mc.push_back({obs.id, x_map, y_map});
    }

    // find landmarks within sensor_range
    std::vector<LandmarkObs> predicted_mc;
    for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
      double distance = dist(p.x, p.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (distance <= sensor_range) {
        predicted_mc.push_back({map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }

    if (predicted_mc.size() > 0) {
      dataAssociation(predicted_mc, observations_mc);

      double particle_weight = 0;

      for (int j=0; j<observations_mc.size(); j++) {
        double weight = multiv_prob(
          std_landmark[0],
          std_landmark[1],
          observations_mc[j].x,
          observations_mc[j].y,
          predicted_mc[observations_mc[j].id].x,
          predicted_mc[observations_mc[j].id].y
        );

        if (particle_weight == 0) {
          particle_weight = weight;
        } else {
          particle_weight *= weight;
        }
      }

      particles[i].weight = particle_weight;
    }

  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::vector<double> weights;
  for (int i=0; i<particles.size(); i++) {
    weights.push_back(particles[i].weight);
  }

  std::default_random_engine gen;
  std::discrete_distribution<int> distribution (weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;

  for (int i=0; i<particles.size(); i++) {
    int n = distribution(gen);
    resampled_particles.push_back(particles[n]);
  }  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}