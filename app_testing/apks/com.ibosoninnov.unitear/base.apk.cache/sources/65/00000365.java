package b.d.b.e1;

/* compiled from: AutoValue_ImmutableZoomState.java */
/* loaded from: classes.dex */
public final class a extends d {

    /* renamed from: a  reason: collision with root package name */
    public final float f1592a;

    /* renamed from: b  reason: collision with root package name */
    public final float f1593b;

    /* renamed from: c  reason: collision with root package name */
    public final float f1594c;

    /* renamed from: d  reason: collision with root package name */
    public final float f1595d;

    public a(float f2, float f3, float f4, float f5) {
        this.f1592a = f2;
        this.f1593b = f3;
        this.f1594c = f4;
        this.f1595d = f5;
    }

    @Override // b.d.b.e1.d
    public float b() {
        return this.f1595d;
    }

    @Override // b.d.b.e1.d
    public float c() {
        return this.f1593b;
    }

    @Override // b.d.b.e1.d
    public float d() {
        return this.f1594c;
    }

    @Override // b.d.b.e1.d
    public float e() {
        return this.f1592a;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof d) {
            d dVar = (d) obj;
            return Float.floatToIntBits(this.f1592a) == Float.floatToIntBits(dVar.e()) && Float.floatToIntBits(this.f1593b) == Float.floatToIntBits(dVar.c()) && Float.floatToIntBits(this.f1594c) == Float.floatToIntBits(dVar.d()) && Float.floatToIntBits(this.f1595d) == Float.floatToIntBits(dVar.b());
        }
        return false;
    }

    public int hashCode() {
        return ((((((Float.floatToIntBits(this.f1592a) ^ 1000003) * 1000003) ^ Float.floatToIntBits(this.f1593b)) * 1000003) ^ Float.floatToIntBits(this.f1594c)) * 1000003) ^ Float.floatToIntBits(this.f1595d);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ImmutableZoomState{zoomRatio=");
        x.append(this.f1592a);
        x.append(", maxZoomRatio=");
        x.append(this.f1593b);
        x.append(", minZoomRatio=");
        x.append(this.f1594c);
        x.append(", linearZoom=");
        x.append(this.f1595d);
        x.append("}");
        return x.toString();
    }
}