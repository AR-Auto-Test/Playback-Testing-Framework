package b.d.b.d1;

import android.util.Size;
import java.util.Objects;

/* compiled from: AutoValue_SurfaceSizeDefinition.java */
/* loaded from: classes.dex */
public final class p extends f1 {

    /* renamed from: a  reason: collision with root package name */
    public final Size f1582a;

    /* renamed from: b  reason: collision with root package name */
    public final Size f1583b;

    /* renamed from: c  reason: collision with root package name */
    public final Size f1584c;

    public p(Size size, Size size2, Size size3) {
        Objects.requireNonNull(size, "Null analysisSize");
        this.f1582a = size;
        Objects.requireNonNull(size2, "Null previewSize");
        this.f1583b = size2;
        Objects.requireNonNull(size3, "Null recordSize");
        this.f1584c = size3;
    }

    @Override // b.d.b.d1.f1
    public Size a() {
        return this.f1582a;
    }

    @Override // b.d.b.d1.f1
    public Size b() {
        return this.f1583b;
    }

    @Override // b.d.b.d1.f1
    public Size c() {
        return this.f1584c;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof f1) {
            f1 f1Var = (f1) obj;
            return this.f1582a.equals(f1Var.a()) && this.f1583b.equals(f1Var.b()) && this.f1584c.equals(f1Var.c());
        }
        return false;
    }

    public int hashCode() {
        return ((((this.f1582a.hashCode() ^ 1000003) * 1000003) ^ this.f1583b.hashCode()) * 1000003) ^ this.f1584c.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("SurfaceSizeDefinition{analysisSize=");
        x.append(this.f1582a);
        x.append(", previewSize=");
        x.append(this.f1583b);
        x.append(", recordSize=");
        x.append(this.f1584c);
        x.append("}");
        return x.toString();
    }
}