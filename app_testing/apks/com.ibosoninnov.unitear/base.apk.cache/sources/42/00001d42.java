package com.google.ar.sceneform.utilities;

/* loaded from: classes.dex */
public class MovingAverage {
    public static final double DEFAULT_WEIGHT = 0.8999999761581421d;
    private double average;
    private final double weight;

    public MovingAverage(double d2) {
        this(d2, 0.8999999761581421d);
    }

    public void addSample(double d2) {
        double d3 = this.weight;
        this.average = ((1.0d - d3) * d2) + (this.average * d3);
    }

    public double getAverage() {
        return this.average;
    }

    public MovingAverage(double d2, double d3) {
        this.average = d2;
        this.weight = d3;
    }
}