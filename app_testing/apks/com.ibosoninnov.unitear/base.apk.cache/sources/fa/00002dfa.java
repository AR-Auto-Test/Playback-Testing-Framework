package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class RotatedRect {
    public double angle;
    public Point center;
    public Size size;

    public RotatedRect() {
        this.center = new Point();
        this.size = new Size();
        this.angle = ShadowDrawableWrapper.COS_45;
    }

    public Rect boundingRect() {
        Point[] pointArr = new Point[4];
        points(pointArr);
        Rect rect = new Rect((int) Math.floor(Math.min(Math.min(Math.min(pointArr[0].x, pointArr[1].x), pointArr[2].x), pointArr[3].x)), (int) Math.floor(Math.min(Math.min(Math.min(pointArr[0].y, pointArr[1].y), pointArr[2].y), pointArr[3].y)), (int) Math.ceil(Math.max(Math.max(Math.max(pointArr[0].x, pointArr[1].x), pointArr[2].x), pointArr[3].x)), (int) Math.ceil(Math.max(Math.max(Math.max(pointArr[0].y, pointArr[1].y), pointArr[2].y), pointArr[3].y)));
        rect.width -= rect.x - 1;
        rect.height -= rect.y - 1;
        return rect;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof RotatedRect) {
            RotatedRect rotatedRect = (RotatedRect) obj;
            return this.center.equals(rotatedRect.center) && this.size.equals(rotatedRect.size) && this.angle == rotatedRect.angle;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.center.x);
        long doubleToLongBits2 = Double.doubleToLongBits(this.center.y);
        int i = ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) (doubleToLongBits2 ^ (doubleToLongBits2 >>> 32)));
        long doubleToLongBits3 = Double.doubleToLongBits(this.size.width);
        int i2 = (i * 31) + ((int) (doubleToLongBits3 ^ (doubleToLongBits3 >>> 32)));
        long doubleToLongBits4 = Double.doubleToLongBits(this.size.height);
        int i3 = (i2 * 31) + ((int) (doubleToLongBits4 ^ (doubleToLongBits4 >>> 32)));
        long doubleToLongBits5 = Double.doubleToLongBits(this.angle);
        return (i3 * 31) + ((int) ((doubleToLongBits5 >>> 32) ^ doubleToLongBits5));
    }

    public void points(Point[] pointArr) {
        double d2 = (this.angle * 3.141592653589793d) / 180.0d;
        double cos = Math.cos(d2) * 0.5d;
        double sin = Math.sin(d2) * 0.5d;
        Point point = this.center;
        double d3 = point.x;
        Size size = this.size;
        double d4 = size.height;
        double d5 = size.width;
        pointArr[0] = new Point((d3 - (sin * d4)) - (cos * d5), ((d4 * cos) + point.y) - (d5 * sin));
        Point point2 = this.center;
        double d6 = point2.x;
        Size size2 = this.size;
        double d7 = size2.height;
        double d8 = (sin * d7) + d6;
        double d9 = size2.width;
        pointArr[1] = new Point(d8 - (cos * d9), (point2.y - (cos * d7)) - (sin * d9));
        Point point3 = this.center;
        pointArr[2] = new Point((point3.x * 2.0d) - pointArr[0].x, (point3.y * 2.0d) - pointArr[0].y);
        Point point4 = this.center;
        pointArr[3] = new Point((point4.x * 2.0d) - pointArr[1].x, (point4.y * 2.0d) - pointArr[1].y);
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            Point point = this.center;
            point.x = dArr.length > 0 ? dArr[0] : 0.0d;
            point.y = dArr.length > 1 ? dArr[1] : 0.0d;
            Size size = this.size;
            size.width = dArr.length > 2 ? dArr[2] : 0.0d;
            size.height = dArr.length > 3 ? dArr[3] : 0.0d;
            if (dArr.length > 4) {
                d2 = dArr[4];
            }
            this.angle = d2;
            return;
        }
        Point point2 = this.center;
        point2.x = ShadowDrawableWrapper.COS_45;
        point2.y = ShadowDrawableWrapper.COS_45;
        Size size2 = this.size;
        size2.width = ShadowDrawableWrapper.COS_45;
        size2.height = ShadowDrawableWrapper.COS_45;
        this.angle = ShadowDrawableWrapper.COS_45;
    }

    public String toString() {
        StringBuilder x = a.x("{ ");
        x.append(this.center);
        x.append(" ");
        x.append(this.size);
        x.append(" * ");
        x.append(this.angle);
        x.append(" }");
        return x.toString();
    }

    /* JADX DEBUG: Method merged with bridge method */
    public RotatedRect clone() {
        return new RotatedRect(this.center, this.size, this.angle);
    }

    public RotatedRect(Point point, Size size, double d2) {
        this.center = point.clone();
        this.size = size.clone();
        this.angle = d2;
    }

    public RotatedRect(double[] dArr) {
        this();
        set(dArr);
    }
}