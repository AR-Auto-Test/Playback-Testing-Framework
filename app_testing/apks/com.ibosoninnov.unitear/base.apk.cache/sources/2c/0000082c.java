package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.os.ParcelFileDescriptor;
import c.c.a.m.x.c.s;
import java.util.Objects;

/* compiled from: ParcelFileDescriptorBitmapDecoder.java */
/* loaded from: classes.dex */
public final class v implements c.c.a.m.r<ParcelFileDescriptor, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final m f4005a;

    public v(m mVar) {
        this.f4005a = mVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(ParcelFileDescriptor parcelFileDescriptor, c.c.a.m.p pVar) {
        Objects.requireNonNull(this.f4005a);
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(ParcelFileDescriptor parcelFileDescriptor, int i, int i2, c.c.a.m.p pVar) {
        m mVar = this.f4005a;
        return mVar.a(new s.b(parcelFileDescriptor, mVar.l, mVar.k), i, i2, pVar, m.f3981f);
    }
}