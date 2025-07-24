package c.e.b;

import android.graphics.drawable.Drawable;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class wd implements c.c.a.q.e<Drawable> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f5383a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Node f5384b;

    public wd(yd ydVar, Node node, Node node2) {
        this.f5383a = node;
        this.f5384b = node2;
    }

    @Override // c.c.a.q.e
    public boolean a(c.c.a.m.v.r rVar, Object obj, c.c.a.q.j.h<Drawable> hVar, boolean z) {
        return false;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object, c.c.a.q.j.h, c.c.a.m.a, boolean] */
    @Override // c.c.a.q.e
    public boolean b(Drawable drawable, Object obj, c.c.a.q.j.h<Drawable> hVar, c.c.a.m.a aVar, boolean z) {
        Drawable drawable2 = drawable;
        float sqrt = (float) Math.sqrt((drawable2.getIntrinsicWidth() / drawable2.getIntrinsicHeight()) * 0.17f);
        this.f5383a.setLocalPosition(new Vector3(-sqrt, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        this.f5384b.setLocalPosition(new Vector3(sqrt, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0.01f));
        return false;
    }
}