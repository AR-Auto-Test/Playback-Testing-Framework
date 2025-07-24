package c.e.b;

import android.graphics.drawable.Drawable;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class ud implements c.c.a.q.e<Drawable> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f5295a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Node f5296b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ vd f5297c;

    public ud(vd vdVar, Node node, Node node2) {
        this.f5297c = vdVar;
        this.f5295a = node;
        this.f5296b = node2;
    }

    @Override // c.c.a.q.e
    public boolean a(c.c.a.m.v.r rVar, Object obj, c.c.a.q.j.h<Drawable> hVar, boolean z) {
        return false;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object, c.c.a.q.j.h, c.c.a.m.a, boolean] */
    @Override // c.c.a.q.e
    public boolean b(Drawable drawable, Object obj, c.c.a.q.j.h<Drawable> hVar, c.c.a.m.a aVar, boolean z) {
        Drawable drawable2 = drawable;
        float intrinsicWidth = drawable2.getIntrinsicWidth() / drawable2.getIntrinsicHeight();
        this.f5297c.f5349h.i = (float) Math.sqrt(intrinsicWidth * 0.17f);
        this.f5295a.setLocalPosition(new Vector3(-this.f5297c.f5349h.i, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        this.f5296b.setLocalPosition(new Vector3(this.f5297c.f5349h.i, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        return false;
    }
}