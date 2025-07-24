package b.h.c;

import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.HashMap;

/* compiled from: ConstraintHelper.java */
/* loaded from: classes.dex */
public abstract class b extends View {

    /* renamed from: b  reason: collision with root package name */
    public int[] f1943b;

    /* renamed from: c  reason: collision with root package name */
    public int f1944c;

    /* renamed from: d  reason: collision with root package name */
    public Context f1945d;

    /* renamed from: e  reason: collision with root package name */
    public b.h.b.i.g f1946e;

    /* renamed from: f  reason: collision with root package name */
    public String f1947f;

    /* renamed from: g  reason: collision with root package name */
    public String f1948g;

    /* renamed from: h  reason: collision with root package name */
    public HashMap<Integer, String> f1949h;

    public b(Context context) {
        super(context);
        this.f1943b = new int[32];
        this.f1949h = new HashMap<>();
        this.f1945d = context;
        f(null);
    }

    public final void a(String str) {
        if (str == null || str.length() == 0 || this.f1945d == null) {
            return;
        }
        String trim = str.trim();
        if (getParent() instanceof ConstraintLayout) {
            ConstraintLayout constraintLayout = (ConstraintLayout) getParent();
        }
        ConstraintLayout constraintLayout2 = getParent() instanceof ConstraintLayout ? (ConstraintLayout) getParent() : null;
        int i = 0;
        if (isInEditMode() && constraintLayout2 != null) {
            Object designInformation = constraintLayout2.getDesignInformation(0, trim);
            if (designInformation instanceof Integer) {
                i = ((Integer) designInformation).intValue();
            }
        }
        if (i == 0 && constraintLayout2 != null) {
            i = e(constraintLayout2, trim);
        }
        if (i == 0) {
            try {
                i = h.class.getField(trim).getInt(null);
            } catch (Exception unused) {
            }
        }
        if (i == 0) {
            i = this.f1945d.getResources().getIdentifier(trim, "id", this.f1945d.getPackageName());
        }
        if (i != 0) {
            this.f1949h.put(Integer.valueOf(i), trim);
            b(i);
            return;
        }
        Log.w("ConstraintHelper", "Could not find id of \"" + trim + "\"");
    }

    public final void b(int i) {
        if (i == getId()) {
            return;
        }
        int i2 = this.f1944c + 1;
        int[] iArr = this.f1943b;
        if (i2 > iArr.length) {
            this.f1943b = Arrays.copyOf(iArr, iArr.length * 2);
        }
        int[] iArr2 = this.f1943b;
        int i3 = this.f1944c;
        iArr2[i3] = i;
        this.f1944c = i3 + 1;
    }

    public final void c(String str) {
        if (str == null || str.length() == 0 || this.f1945d == null) {
            return;
        }
        String trim = str.trim();
        ConstraintLayout constraintLayout = getParent() instanceof ConstraintLayout ? (ConstraintLayout) getParent() : null;
        if (constraintLayout == null) {
            Log.w("ConstraintHelper", "Parent not a ConstraintLayout");
            return;
        }
        int childCount = constraintLayout.getChildCount();
        for (int i = 0; i < childCount; i++) {
            View childAt = constraintLayout.getChildAt(i);
            ViewGroup.LayoutParams layoutParams = childAt.getLayoutParams();
            if ((layoutParams instanceof ConstraintLayout.a) && trim.equals(((ConstraintLayout.a) layoutParams).U)) {
                if (childAt.getId() == -1) {
                    StringBuilder x = c.b.a.a.a.x("to use ConstraintTag view ");
                    x.append(childAt.getClass().getSimpleName());
                    x.append(" must have an ID");
                    Log.w("ConstraintHelper", x.toString());
                } else {
                    b(childAt.getId());
                }
            }
        }
    }

    public void d() {
        ViewParent parent = getParent();
        if (parent == null || !(parent instanceof ConstraintLayout)) {
            return;
        }
        ConstraintLayout constraintLayout = (ConstraintLayout) parent;
        int visibility = getVisibility();
        float elevation = getElevation();
        for (int i = 0; i < this.f1944c; i++) {
            View viewById = constraintLayout.getViewById(this.f1943b[i]);
            if (viewById != null) {
                viewById.setVisibility(visibility);
                if (elevation > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    viewById.setTranslationZ(viewById.getTranslationZ() + elevation);
                }
            }
        }
    }

    public final int e(ConstraintLayout constraintLayout, String str) {
        Resources resources;
        if (str == null || (resources = this.f1945d.getResources()) == null) {
            return 0;
        }
        int childCount = constraintLayout.getChildCount();
        for (int i = 0; i < childCount; i++) {
            View childAt = constraintLayout.getChildAt(i);
            if (childAt.getId() != -1) {
                String str2 = null;
                try {
                    str2 = resources.getResourceEntryName(childAt.getId());
                } catch (Resources.NotFoundException unused) {
                }
                if (str.equals(str2)) {
                    return childAt.getId();
                }
            }
        }
        return 0;
    }

    public void f(AttributeSet attributeSet) {
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, i.f2010b);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 19) {
                    String string = obtainStyledAttributes.getString(index);
                    this.f1947f = string;
                    setIds(string);
                } else if (index == 20) {
                    String string2 = obtainStyledAttributes.getString(index);
                    this.f1948g = string2;
                    setReferenceTags(string2);
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    public void g(b.h.b.i.d dVar, boolean z) {
    }

    public int[] getReferencedIds() {
        return Arrays.copyOf(this.f1943b, this.f1944c);
    }

    public void h() {
    }

    public void i() {
    }

    public void j() {
    }

    public void k() {
        if (this.f1946e == null) {
            return;
        }
        ViewGroup.LayoutParams layoutParams = getLayoutParams();
        if (layoutParams instanceof ConstraintLayout.a) {
            ((ConstraintLayout.a) layoutParams).l0 = (b.h.b.i.d) this.f1946e;
        }
    }

    @Override // android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        String str = this.f1947f;
        if (str != null) {
            setIds(str);
        }
        String str2 = this.f1948g;
        if (str2 != null) {
            setReferenceTags(str2);
        }
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        setMeasuredDimension(0, 0);
    }

    public void setIds(String str) {
        this.f1947f = str;
        if (str == null) {
            return;
        }
        int i = 0;
        this.f1944c = 0;
        while (true) {
            int indexOf = str.indexOf(44, i);
            if (indexOf == -1) {
                a(str.substring(i));
                return;
            } else {
                a(str.substring(i, indexOf));
                i = indexOf + 1;
            }
        }
    }

    public void setReferenceTags(String str) {
        this.f1948g = str;
        if (str == null) {
            return;
        }
        int i = 0;
        this.f1944c = 0;
        while (true) {
            int indexOf = str.indexOf(44, i);
            if (indexOf == -1) {
                c(str.substring(i));
                return;
            } else {
                c(str.substring(i, indexOf));
                i = indexOf + 1;
            }
        }
    }

    public void setReferencedIds(int[] iArr) {
        this.f1947f = null;
        this.f1944c = 0;
        for (int i : iArr) {
            b(i);
        }
    }

    @Override // android.view.View
    public void setTag(int i, Object obj) {
        super.setTag(i, obj);
        if (obj == null && this.f1947f == null) {
            b(i);
        }
    }

    public b(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.f1943b = new int[32];
        this.f1949h = new HashMap<>();
        this.f1945d = context;
        f(attributeSet);
    }
}