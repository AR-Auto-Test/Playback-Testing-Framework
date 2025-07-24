package c.a.a.z;

import b.d.b.m0;

/* compiled from: DocumentData.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3261a;

    /* renamed from: b  reason: collision with root package name */
    public final String f3262b;

    /* renamed from: c  reason: collision with root package name */
    public final float f3263c;

    /* renamed from: d  reason: collision with root package name */
    public final int f3264d;

    /* renamed from: e  reason: collision with root package name */
    public final int f3265e;

    /* renamed from: f  reason: collision with root package name */
    public final float f3266f;

    /* renamed from: g  reason: collision with root package name */
    public final float f3267g;

    /* renamed from: h  reason: collision with root package name */
    public final int f3268h;
    public final int i;
    public final float j;
    public final boolean k;

    public b(String str, String str2, float f2, int i, int i2, float f3, float f4, int i3, int i4, float f5, boolean z) {
        this.f3261a = str;
        this.f3262b = str2;
        this.f3263c = f2;
        this.f3264d = i;
        this.f3265e = i2;
        this.f3266f = f3;
        this.f3267g = f4;
        this.f3268h = i3;
        this.i = i4;
        this.j = f5;
        this.k = z;
    }

    public int hashCode() {
        int hashCode = this.f3262b.hashCode();
        int f2 = ((m0.f(this.f3264d) + (((int) (((hashCode + (this.f3261a.hashCode() * 31)) * 31) + this.f3263c)) * 31)) * 31) + this.f3265e;
        long floatToRawIntBits = Float.floatToRawIntBits(this.f3266f);
        return (((f2 * 31) + ((int) (floatToRawIntBits ^ (floatToRawIntBits >>> 32)))) * 31) + this.f3268h;
    }
}