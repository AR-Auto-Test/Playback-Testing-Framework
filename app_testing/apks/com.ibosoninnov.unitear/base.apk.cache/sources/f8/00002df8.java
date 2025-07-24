package org.opencv.core;

import c.b.a.a.a;

/* loaded from: classes2.dex */
public class Rect {
    public int height;
    public int width;
    public int x;
    public int y;

    public Rect(int i, int i2, int i3, int i4) {
        this.x = i;
        this.y = i2;
        this.width = i3;
        this.height = i4;
    }

    public double area() {
        return this.width * this.height;
    }

    public Point br() {
        return new Point(this.x + this.width, this.y + this.height);
    }

    public boolean contains(Point point) {
        int i = this.x;
        double d2 = point.x;
        if (i <= d2 && d2 < i + this.width) {
            int i2 = this.y;
            double d3 = point.y;
            if (i2 <= d3 && d3 < i2 + this.height) {
                return true;
            }
        }
        return false;
    }

    public boolean empty() {
        return this.width <= 0 || this.height <= 0;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof Rect) {
            Rect rect = (Rect) obj;
            return this.x == rect.x && this.y == rect.y && this.width == rect.width && this.height == rect.height;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.height);
        long doubleToLongBits2 = Double.doubleToLongBits(this.width);
        int i = ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) (doubleToLongBits2 ^ (doubleToLongBits2 >>> 32)));
        long doubleToLongBits3 = Double.doubleToLongBits(this.x);
        int i2 = (i * 31) + ((int) (doubleToLongBits3 ^ (doubleToLongBits3 >>> 32)));
        long doubleToLongBits4 = Double.doubleToLongBits(this.y);
        return (i2 * 31) + ((int) ((doubleToLongBits4 >>> 32) ^ doubleToLongBits4));
    }

    public void set(double[] dArr) {
        if (dArr != null) {
            this.x = dArr.length > 0 ? (int) dArr[0] : 0;
            this.y = dArr.length > 1 ? (int) dArr[1] : 0;
            this.width = dArr.length > 2 ? (int) dArr[2] : 0;
            this.height = dArr.length > 3 ? (int) dArr[3] : 0;
            return;
        }
        this.x = 0;
        this.y = 0;
        this.width = 0;
        this.height = 0;
    }

    public Size size() {
        return new Size(this.width, this.height);
    }

    public Point tl() {
        return new Point(this.x, this.y);
    }

    public String toString() {
        StringBuilder x = a.x("{");
        x.append(this.x);
        x.append(", ");
        x.append(this.y);
        x.append(", ");
        x.append(this.width);
        x.append("x");
        return a.s(x, this.height, "}");
    }

    /* JADX DEBUG: Method merged with bridge method */
    public Rect clone() {
        return new Rect(this.x, this.y, this.width, this.height);
    }

    public Rect() {
        this(0, 0, 0, 0);
    }

    public Rect(Point point, Point point2) {
        double d2 = point.x;
        double d3 = point2.x;
        int i = (int) (d2 < d3 ? d2 : d3);
        this.x = i;
        double d4 = point.y;
        double d5 = point2.y;
        int i2 = (int) (d4 < d5 ? d4 : d5);
        this.y = i2;
        this.width = ((int) (d2 <= d3 ? d3 : d2)) - i;
        this.height = ((int) (d4 <= d5 ? d5 : d4)) - i2;
    }

    public Rect(Point point, Size size) {
        this((int) point.x, (int) point.y, (int) size.width, (int) size.height);
    }

    public Rect(double[] dArr) {
        set(dArr);
    }
}