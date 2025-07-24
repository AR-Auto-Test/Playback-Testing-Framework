package c.a.a.b0;

import android.graphics.Color;
import android.graphics.PointF;
import c.a.a.b0.h0.c;
import java.util.ArrayList;

/* compiled from: GradientColorParser.java */
/* loaded from: classes.dex */
public class k implements g0<c.a.a.z.k.c> {

    /* renamed from: a  reason: collision with root package name */
    public int f2992a;

    public k(int i) {
        this.f2992a = i;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public c.a.a.z.k.c a(c.a.a.b0.h0.c cVar, float f2) {
        int i;
        double d2;
        ArrayList arrayList = new ArrayList();
        int i2 = 0;
        boolean z = cVar.M() == c.b.BEGIN_ARRAY;
        if (z) {
            cVar.B();
        }
        while (cVar.G()) {
            arrayList.add(Float.valueOf((float) cVar.I()));
        }
        if (z) {
            cVar.D();
        }
        if (this.f2992a == -1) {
            this.f2992a = arrayList.size() / 4;
        }
        int i3 = this.f2992a;
        float[] fArr = new float[i3];
        int[] iArr = new int[i3];
        int i4 = 0;
        int i5 = 0;
        int i6 = 0;
        while (true) {
            i = this.f2992a * 4;
            if (i4 >= i) {
                break;
            }
            int i7 = i4 / 4;
            double floatValue = ((Float) arrayList.get(i4)).floatValue();
            int i8 = i4 % 4;
            if (i8 == 0) {
                fArr[i7] = (float) floatValue;
            } else if (i8 == 1) {
                i5 = (int) (floatValue * 255.0d);
            } else if (i8 == 2) {
                i6 = (int) (floatValue * 255.0d);
            } else if (i8 == 3) {
                iArr[i7] = Color.argb(255, i5, i6, (int) (floatValue * 255.0d));
            }
            i4++;
        }
        c.a.a.z.k.c cVar2 = new c.a.a.z.k.c(fArr, iArr);
        if (arrayList.size() > i) {
            int size = (arrayList.size() - i) / 2;
            double[] dArr = new double[size];
            double[] dArr2 = new double[size];
            int i9 = 0;
            while (i < arrayList.size()) {
                if (i % 2 == 0) {
                    dArr[i9] = ((Float) arrayList.get(i)).floatValue();
                } else {
                    dArr2[i9] = ((Float) arrayList.get(i)).floatValue();
                    i9++;
                }
                i++;
            }
            while (true) {
                int[] iArr2 = cVar2.f3308b;
                if (i2 >= iArr2.length) {
                    break;
                }
                int i10 = iArr2[i2];
                double d3 = cVar2.f3307a[i2];
                int i11 = 1;
                while (true) {
                    if (i11 < size) {
                        int i12 = i11 - 1;
                        double d4 = dArr[i12];
                        double d5 = dArr[i11];
                        if (dArr[i11] >= d3) {
                            double d6 = dArr2[i12];
                            double d7 = dArr2[i11];
                            PointF pointF = c.a.a.c0.f.f3030a;
                            d2 = (((d7 - d6) * ((d3 - d4) / (d5 - d4))) + d6) * 255.0d;
                            break;
                        }
                        i11++;
                    } else {
                        d2 = dArr2[size - 1] * 255.0d;
                        break;
                    }
                }
                cVar2.f3308b[i2] = Color.argb((int) d2, Color.red(i10), Color.green(i10), Color.blue(i10));
                i2++;
            }
        }
        return cVar2;
    }
}