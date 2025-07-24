package c.c.a.m.x.e;

import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import c.c.a.m.v.s;
import c.c.a.m.v.w;
import java.util.Objects;

/* compiled from: DrawableResource.java */
/* loaded from: classes.dex */
public abstract class b<T extends Drawable> implements w<T>, s {

    /* renamed from: b  reason: collision with root package name */
    public final T f4023b;

    public b(T t) {
        Objects.requireNonNull(t, "Argument must not be null");
        this.f4023b = t;
    }

    @Override // c.c.a.m.v.w
    public Object get() {
        Drawable.ConstantState constantState = this.f4023b.getConstantState();
        if (constantState == null) {
            return this.f4023b;
        }
        return constantState.newDrawable();
    }

    @Override // c.c.a.m.v.s
    public void initialize() {
        T t = this.f4023b;
        if (t instanceof BitmapDrawable) {
            ((BitmapDrawable) t).getBitmap().prepareToDraw();
        } else if (t instanceof c.c.a.m.x.g.c) {
            ((c.c.a.m.x.g.c) t).b().prepareToDraw();
        }
    }
}