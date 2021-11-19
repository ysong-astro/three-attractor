#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 08:26:56 2020

@author: spacedog
"""
#this script is used to calculate the corrected cosmic velocity of an object using a 3-attractor flow model described in Mould+2000
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from cosmo_calculator import cosmology

#sun's motion relative to local group center
Vlg=np.array([-79, 296, -36]) #velocity

#motion directions
v_virgo_h=SkyCoord('12h28m19s','+12d40m',frame='fk5',equinox='J1950.0').galactic.cartesian.xyz.value
v_ga_h=SkyCoord('13h20m00s','-44d00m',frame='fk5',equinox='J1950.0').galactic.cartesian.xyz.value
v_shapley_h=SkyCoord('13h30m','-31d00m',frame='fk5',equinox='J1950.0').galactic.cartesian.xyz.value
c=299792.458 #km/s light speed

def V_infall(Vfid, va, v_h, v_gal_h,v0):
    cost=np.dot(v_h,v_gal_h)/(np.linalg.norm(v_h)*np.linalg.norm(v_gal_h))
    roa2=v0**2+va**2-2*v0*va*cost
    vin=Vfid*cost+Vfid*va*(v0-va*cost)/roa2
    return vin
    
#3-attractor model Mould+2000
def V_cosmic(V_H,l,b):
    v_gal_h=SkyCoord(l=l*u.degree,b=b*u.degree,frame='galactic').cartesian.xyz.value
    l_rad=np.radians(l)
    b_rad=np.radians(b)
    v0=V_H-79*np.cos(l_rad)*np.cos(b_rad)+296*np.sin(l_rad)*np.cos(b_rad)-36*np.sin(b_rad)
    vin_virgo=V_infall(200,1035,v_virgo_h,v_gal_h,v0)
    vin_ga=V_infall(400,4600,v_ga_h,v_gal_h,v0)
    vin_shapley=V_infall(85,13800,v_shapley_h,v_gal_h,v0)
    V_cosmic=v0+vin_virgo+vin_ga+vin_shapley
    return (V_H, v0, vin_virgo, vin_ga, vin_shapley, V_cosmic) #heliocentric, LG corrected, Virgo corrected, GA corrected, Shapley corrected, final
    
def dl_da(V_H,l,b):
    #use Armus+2009 cosmology
    H0=70.0
    ome_m=0.28
    ome_vac=0.72
    z=V_cosmic(V_H,l,b)[-1]/c #3-attractor corrected velocity
    DL,scale=cosmology(z,H0,ome_m,ome_vac)
    return (DL,scale)
