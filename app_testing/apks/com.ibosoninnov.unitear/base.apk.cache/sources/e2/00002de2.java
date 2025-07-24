package org.opencv.core;

import c.b.a.a.a;

/* loaded from: classes2.dex */
public class DMatch {
    public float distance;
    public int imgIdx;
    public int queryIdx;
    public int trainIdx;

    public DMatch() {
        this(-1, -1, Float.MAX_VALUE);
    }

    public boolean lessThan(DMatch dMatch) {
        return this.distance < dMatch.distance;
    }

    public String toString() {
        StringBuilder x = a.x("DMatch [queryIdx=");
        x.append(this.queryIdx);
        x.append(", trainIdx=");
        x.append(this.trainIdx);
        x.append(", imgIdx=");
        x.append(this.imgIdx);
        x.append(", distance=");
        x.append(this.distance);
        x.append("]");
        return x.toString();
    }

    public DMatch(int i, int i2, float f2) {
        this.queryIdx = i;
        this.trainIdx = i2;
        this.imgIdx = -1;
        this.distance = f2;
    }

    public DMatch(int i, int i2, int i3, float f2) {
        this.queryIdx = i;
        this.trainIdx = i2;
        this.imgIdx = i3;
        this.distance = f2;
    }
}