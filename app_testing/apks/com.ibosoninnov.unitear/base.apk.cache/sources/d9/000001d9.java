package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.widget.MultiAutoCompleteTextView;
import com.ibosoninnov.unitear.R;

/* compiled from: AppCompatMultiAutoCompleteTextView.java */
/* loaded from: classes.dex */
public class o extends MultiAutoCompleteTextView {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f894b = {16843126};

    /* renamed from: c  reason: collision with root package name */
    public final e f895c;

    /* renamed from: d  reason: collision with root package name */
    public final y f896d;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public o(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, R.attr.autoCompleteTextViewStyle);
        v0.a(context);
        t0.a(this, getContext());
        y0 r = y0.r(getContext(), attributeSet, f894b, R.attr.autoCompleteTextViewStyle, 0);
        if (r.p(0)) {
            setDropDownBackgroundDrawable(r.g(0));
        }
        r.f972b.recycle();
        e eVar = new e(this);
        this.f895c = eVar;
        eVar.d(attributeSet, R.attr.autoCompleteTextViewStyle);
        y yVar = new y(this);
        this.f896d = yVar;
        yVar.e(attributeSet, R.attr.autoCompleteTextViewStyle);
        yVar.b();
    }

    @Override // android.widget.TextView, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        e eVar = this.f895c;
        if (eVar != null) {
            eVar.a();
        }
        y yVar = this.f896d;
        if (yVar != null) {
            yVar.b();
        }
    }

    public ColorStateList getSupportBackgroundTintList() {
        e eVar = this.f895c;
        if (eVar != null) {
            return eVar.b();
        }
        return null;
    }

    public PorterDuff.Mode getSupportBackgroundTintMode() {
        e eVar = this.f895c;
        if (eVar != null) {
            return eVar.c();
        }
        return null;
    }

    @Override // android.widget.TextView, android.view.View
    public InputConnection onCreateInputConnection(EditorInfo editorInfo) {
        InputConnection onCreateInputConnection = super.onCreateInputConnection(editorInfo);
        b.b.a.m(onCreateInputConnection, editorInfo, this);
        return onCreateInputConnection;
    }

    @Override // android.view.View
    public void setBackgroundDrawable(Drawable drawable) {
        super.setBackgroundDrawable(drawable);
        e eVar = this.f895c;
        if (eVar != null) {
            eVar.e();
        }
    }

    @Override // android.view.View
    public void setBackgroundResource(int i) {
        super.setBackgroundResource(i);
        e eVar = this.f895c;
        if (eVar != null) {
            eVar.f(i);
        }
    }

    @Override // android.widget.AutoCompleteTextView
    public void setDropDownBackgroundResource(int i) {
        setDropDownBackgroundDrawable(b.b.d.a.a.a(getContext(), i));
    }

    public void setSupportBackgroundTintList(ColorStateList colorStateList) {
        e eVar = this.f895c;
        if (eVar != null) {
            eVar.h(colorStateList);
        }
    }

    public void setSupportBackgroundTintMode(PorterDuff.Mode mode) {
        e eVar = this.f895c;
        if (eVar != null) {
            eVar.i(mode);
        }
    }

    @Override // android.widget.TextView
    public void setTextAppearance(Context context, int i) {
        super.setTextAppearance(context, i);
        y yVar = this.f896d;
        if (yVar != null) {
            yVar.f(context, i);
        }
    }
}