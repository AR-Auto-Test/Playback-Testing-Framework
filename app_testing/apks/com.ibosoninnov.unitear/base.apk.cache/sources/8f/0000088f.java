package c.c.a.q.j;

import android.content.Context;
import android.graphics.Point;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/* compiled from: ViewTarget.java */
@Deprecated
/* loaded from: classes.dex */
public abstract class i<T extends View, Z> extends c.c.a.q.j.a<Z> {

    /* renamed from: b  reason: collision with root package name */
    public final T f4158b;

    /* renamed from: c  reason: collision with root package name */
    public final a f4159c;

    /* compiled from: ViewTarget.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public static Integer f4160a;

        /* renamed from: b  reason: collision with root package name */
        public final View f4161b;

        /* renamed from: c  reason: collision with root package name */
        public final List<g> f4162c = new ArrayList();

        /* renamed from: d  reason: collision with root package name */
        public ViewTreeObserver$OnPreDrawListenerC0084a f4163d;

        /* compiled from: ViewTarget.java */
        /* renamed from: c.c.a.q.j.i$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public static final class ViewTreeObserver$OnPreDrawListenerC0084a implements ViewTreeObserver.OnPreDrawListener {

            /* renamed from: b  reason: collision with root package name */
            public final WeakReference<a> f4164b;

            public ViewTreeObserver$OnPreDrawListenerC0084a(a aVar) {
                this.f4164b = new WeakReference<>(aVar);
            }

            @Override // android.view.ViewTreeObserver.OnPreDrawListener
            public boolean onPreDraw() {
                if (Log.isLoggable("ViewTarget", 2)) {
                    Log.v("ViewTarget", "OnGlobalLayoutListener called attachStateListener=" + this);
                }
                a aVar = this.f4164b.get();
                if (aVar == null || aVar.f4162c.isEmpty()) {
                    return true;
                }
                int d2 = aVar.d();
                int c2 = aVar.c();
                if (aVar.e(d2, c2)) {
                    Iterator it = new ArrayList(aVar.f4162c).iterator();
                    while (it.hasNext()) {
                        ((g) it.next()).b(d2, c2);
                    }
                    aVar.a();
                    return true;
                }
                return true;
            }
        }

        public a(View view) {
            this.f4161b = view;
        }

        public void a() {
            ViewTreeObserver viewTreeObserver = this.f4161b.getViewTreeObserver();
            if (viewTreeObserver.isAlive()) {
                viewTreeObserver.removeOnPreDrawListener(this.f4163d);
            }
            this.f4163d = null;
            this.f4162c.clear();
        }

        public final int b(int i, int i2, int i3) {
            int i4 = i2 - i3;
            if (i4 > 0) {
                return i4;
            }
            int i5 = i - i3;
            if (i5 > 0) {
                return i5;
            }
            if (this.f4161b.isLayoutRequested() || i2 != -2) {
                return 0;
            }
            if (Log.isLoggable("ViewTarget", 4)) {
                Log.i("ViewTarget", "Glide treats LayoutParams.WRAP_CONTENT as a request for an image the size of this device's screen dimensions. If you want to load the original image and are ok with the corresponding memory cost and OOMs (depending on the input size), use override(Target.SIZE_ORIGINAL). Otherwise, use LayoutParams.MATCH_PARENT, set layout_width and layout_height to fixed dimension, or use .override() with fixed dimensions.");
            }
            Context context = this.f4161b.getContext();
            if (f4160a == null) {
                WindowManager windowManager = (WindowManager) context.getSystemService("window");
                Objects.requireNonNull(windowManager, "Argument must not be null");
                Display defaultDisplay = windowManager.getDefaultDisplay();
                Point point = new Point();
                defaultDisplay.getSize(point);
                f4160a = Integer.valueOf(Math.max(point.x, point.y));
            }
            return f4160a.intValue();
        }

        public final int c() {
            int paddingBottom = this.f4161b.getPaddingBottom() + this.f4161b.getPaddingTop();
            ViewGroup.LayoutParams layoutParams = this.f4161b.getLayoutParams();
            return b(this.f4161b.getHeight(), layoutParams != null ? layoutParams.height : 0, paddingBottom);
        }

        public final int d() {
            int paddingRight = this.f4161b.getPaddingRight() + this.f4161b.getPaddingLeft();
            ViewGroup.LayoutParams layoutParams = this.f4161b.getLayoutParams();
            return b(this.f4161b.getWidth(), layoutParams != null ? layoutParams.width : 0, paddingRight);
        }

        public final boolean e(int i, int i2) {
            if (i > 0 || i == Integer.MIN_VALUE) {
                return i2 > 0 || i2 == Integer.MIN_VALUE;
            }
            return false;
        }
    }

    public i(T t) {
        Objects.requireNonNull(t, "Argument must not be null");
        this.f4158b = t;
        this.f4159c = new a(t);
    }

    @Override // c.c.a.q.j.h
    public void a(g gVar) {
        this.f4159c.f4162c.remove(gVar);
    }

    @Override // c.c.a.q.j.h
    public void c(c.c.a.q.c cVar) {
        this.f4158b.setTag(R.id.glide_custom_view_target_tag, cVar);
    }

    @Override // c.c.a.q.j.h
    public c.c.a.q.c f() {
        Object tag = this.f4158b.getTag(R.id.glide_custom_view_target_tag);
        if (tag != null) {
            if (tag instanceof c.c.a.q.c) {
                return (c.c.a.q.c) tag;
            }
            throw new IllegalArgumentException("You must not call setTag() on a view Glide is targeting");
        }
        return null;
    }

    @Override // c.c.a.q.j.h
    public void h(g gVar) {
        a aVar = this.f4159c;
        int d2 = aVar.d();
        int c2 = aVar.c();
        if (aVar.e(d2, c2)) {
            ((c.c.a.q.h) gVar).b(d2, c2);
            return;
        }
        if (!aVar.f4162c.contains(gVar)) {
            aVar.f4162c.add(gVar);
        }
        if (aVar.f4163d == null) {
            ViewTreeObserver viewTreeObserver = aVar.f4161b.getViewTreeObserver();
            a.ViewTreeObserver$OnPreDrawListenerC0084a viewTreeObserver$OnPreDrawListenerC0084a = new a.ViewTreeObserver$OnPreDrawListenerC0084a(aVar);
            aVar.f4163d = viewTreeObserver$OnPreDrawListenerC0084a;
            viewTreeObserver.addOnPreDrawListener(viewTreeObserver$OnPreDrawListenerC0084a);
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Target for: ");
        x.append(this.f4158b);
        return x.toString();
    }
}