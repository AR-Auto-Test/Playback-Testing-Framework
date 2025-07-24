package androidx.appcompat.view.menu;

import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.TextView;
import b.b.b;
import b.b.g.i.i;
import b.b.g.i.n;
import b.b.h.y0;
import b.j.j.q;
import com.ibosoninnov.unitear.R;
import java.util.concurrent.atomic.AtomicInteger;

/* loaded from: classes.dex */
public class ListMenuItemView extends LinearLayout implements n.a, AbsListView.SelectionBoundsAdjuster {

    /* renamed from: b  reason: collision with root package name */
    public i f88b;

    /* renamed from: c  reason: collision with root package name */
    public ImageView f89c;

    /* renamed from: d  reason: collision with root package name */
    public RadioButton f90d;

    /* renamed from: e  reason: collision with root package name */
    public TextView f91e;

    /* renamed from: f  reason: collision with root package name */
    public CheckBox f92f;

    /* renamed from: g  reason: collision with root package name */
    public TextView f93g;

    /* renamed from: h  reason: collision with root package name */
    public ImageView f94h;
    public ImageView i;
    public LinearLayout j;
    public Drawable k;
    public int l;
    public Context m;
    public boolean n;
    public Drawable o;
    public boolean p;
    public LayoutInflater q;
    public boolean r;

    public ListMenuItemView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        y0 r = y0.r(getContext(), attributeSet, b.r, R.attr.listMenuViewStyle, 0);
        this.k = r.g(5);
        this.l = r.m(1, -1);
        this.n = r.a(7, false);
        this.m = context;
        this.o = r.g(8);
        TypedArray obtainStyledAttributes = context.getTheme().obtainStyledAttributes(null, new int[]{16843049}, R.attr.dropDownListViewStyle, 0);
        this.p = obtainStyledAttributes.hasValue(0);
        r.f972b.recycle();
        obtainStyledAttributes.recycle();
    }

    private LayoutInflater getInflater() {
        if (this.q == null) {
            this.q = LayoutInflater.from(getContext());
        }
        return this.q;
    }

    private void setSubMenuArrowVisible(boolean z) {
        ImageView imageView = this.f94h;
        if (imageView != null) {
            imageView.setVisibility(z ? 0 : 8);
        }
    }

    public final void a() {
        CheckBox checkBox = (CheckBox) getInflater().inflate(R.layout.abc_list_menu_item_checkbox, (ViewGroup) this, false);
        this.f92f = checkBox;
        LinearLayout linearLayout = this.j;
        if (linearLayout != null) {
            linearLayout.addView(checkBox, -1);
        } else {
            addView(checkBox, -1);
        }
    }

    @Override // android.widget.AbsListView.SelectionBoundsAdjuster
    public void adjustListItemSelectionBounds(Rect rect) {
        ImageView imageView = this.i;
        if (imageView == null || imageView.getVisibility() != 0) {
            return;
        }
        LinearLayout.LayoutParams layoutParams = (LinearLayout.LayoutParams) this.i.getLayoutParams();
        rect.top = this.i.getHeight() + layoutParams.topMargin + layoutParams.bottomMargin + rect.top;
    }

    public final void b() {
        RadioButton radioButton = (RadioButton) getInflater().inflate(R.layout.abc_list_menu_item_radio, (ViewGroup) this, false);
        this.f90d = radioButton;
        LinearLayout linearLayout = this.j;
        if (linearLayout != null) {
            linearLayout.addView(radioButton, -1);
        } else {
            addView(radioButton, -1);
        }
    }

    public void c(boolean z) {
        String sb;
        int i = (z && this.f88b.n()) ? 0 : 8;
        if (i == 0) {
            TextView textView = this.f93g;
            i iVar = this.f88b;
            char e2 = iVar.e();
            if (e2 == 0) {
                sb = "";
            } else {
                Resources resources = iVar.n.getContext().getResources();
                StringBuilder sb2 = new StringBuilder();
                if (ViewConfiguration.get(iVar.n.getContext()).hasPermanentMenuKey()) {
                    sb2.append(resources.getString(R.string.abc_prepend_shortcut_label));
                }
                int i2 = iVar.n.isQwertyMode() ? iVar.k : iVar.i;
                i.c(sb2, i2, 65536, resources.getString(R.string.abc_menu_meta_shortcut_label));
                i.c(sb2, i2, 4096, resources.getString(R.string.abc_menu_ctrl_shortcut_label));
                i.c(sb2, i2, 2, resources.getString(R.string.abc_menu_alt_shortcut_label));
                i.c(sb2, i2, 1, resources.getString(R.string.abc_menu_shift_shortcut_label));
                i.c(sb2, i2, 4, resources.getString(R.string.abc_menu_sym_shortcut_label));
                i.c(sb2, i2, 8, resources.getString(R.string.abc_menu_function_shortcut_label));
                if (e2 == '\b') {
                    sb2.append(resources.getString(R.string.abc_menu_delete_shortcut_label));
                } else if (e2 == '\n') {
                    sb2.append(resources.getString(R.string.abc_menu_enter_shortcut_label));
                } else if (e2 != ' ') {
                    sb2.append(e2);
                } else {
                    sb2.append(resources.getString(R.string.abc_menu_space_shortcut_label));
                }
                sb = sb2.toString();
            }
            textView.setText(sb);
        }
        if (this.f93g.getVisibility() != i) {
            this.f93g.setVisibility(i);
        }
    }

    @Override // b.b.g.i.n.a
    public i getItemData() {
        return this.f88b;
    }

    @Override // b.b.g.i.n.a
    public void initialize(i iVar, int i) {
        this.f88b = iVar;
        setVisibility(iVar.isVisible() ? 0 : 8);
        setTitle(iVar.f734e);
        setCheckable(iVar.isCheckable());
        boolean n = iVar.n();
        iVar.e();
        c(n);
        setIcon(iVar.getIcon());
        setEnabled(iVar.isEnabled());
        setSubMenuArrowVisible(iVar.hasSubMenu());
        setContentDescription(iVar.q);
    }

    @Override // android.view.View
    public void onFinishInflate() {
        super.onFinishInflate();
        Drawable drawable = this.k;
        AtomicInteger atomicInteger = q.f2214a;
        setBackground(drawable);
        TextView textView = (TextView) findViewById(R.id.title);
        this.f91e = textView;
        int i = this.l;
        if (i != -1) {
            textView.setTextAppearance(this.m, i);
        }
        this.f93g = (TextView) findViewById(R.id.shortcut);
        ImageView imageView = (ImageView) findViewById(R.id.submenuarrow);
        this.f94h = imageView;
        if (imageView != null) {
            imageView.setImageDrawable(this.o);
        }
        this.i = (ImageView) findViewById(R.id.group_divider);
        this.j = (LinearLayout) findViewById(R.id.content);
    }

    @Override // android.widget.LinearLayout, android.view.View
    public void onMeasure(int i, int i2) {
        if (this.f89c != null && this.n) {
            ViewGroup.LayoutParams layoutParams = getLayoutParams();
            LinearLayout.LayoutParams layoutParams2 = (LinearLayout.LayoutParams) this.f89c.getLayoutParams();
            int i3 = layoutParams.height;
            if (i3 > 0 && layoutParams2.width <= 0) {
                layoutParams2.width = i3;
            }
        }
        super.onMeasure(i, i2);
    }

    public void setCheckable(boolean z) {
        CompoundButton compoundButton;
        CompoundButton compoundButton2;
        if (!z && this.f90d == null && this.f92f == null) {
            return;
        }
        if (this.f88b.h()) {
            if (this.f90d == null) {
                b();
            }
            compoundButton = this.f90d;
            compoundButton2 = this.f92f;
        } else {
            if (this.f92f == null) {
                a();
            }
            compoundButton = this.f92f;
            compoundButton2 = this.f90d;
        }
        if (z) {
            compoundButton.setChecked(this.f88b.isChecked());
            if (compoundButton.getVisibility() != 0) {
                compoundButton.setVisibility(0);
            }
            if (compoundButton2 == null || compoundButton2.getVisibility() == 8) {
                return;
            }
            compoundButton2.setVisibility(8);
            return;
        }
        CheckBox checkBox = this.f92f;
        if (checkBox != null) {
            checkBox.setVisibility(8);
        }
        RadioButton radioButton = this.f90d;
        if (radioButton != null) {
            radioButton.setVisibility(8);
        }
    }

    public void setChecked(boolean z) {
        CompoundButton compoundButton;
        if (this.f88b.h()) {
            if (this.f90d == null) {
                b();
            }
            compoundButton = this.f90d;
        } else {
            if (this.f92f == null) {
                a();
            }
            compoundButton = this.f92f;
        }
        compoundButton.setChecked(z);
    }

    public void setForceShowIcon(boolean z) {
        this.r = z;
        this.n = z;
    }

    public void setGroupDividerEnabled(boolean z) {
        ImageView imageView = this.i;
        if (imageView != null) {
            imageView.setVisibility((this.p || !z) ? 8 : 0);
        }
    }

    public void setIcon(Drawable drawable) {
        boolean z = this.f88b.n.getOptionalIconsVisible() || this.r;
        if (z || this.n) {
            ImageView imageView = this.f89c;
            if (imageView == null && drawable == null && !this.n) {
                return;
            }
            if (imageView == null) {
                ImageView imageView2 = (ImageView) getInflater().inflate(R.layout.abc_list_menu_item_icon, (ViewGroup) this, false);
                this.f89c = imageView2;
                LinearLayout linearLayout = this.j;
                if (linearLayout != null) {
                    linearLayout.addView(imageView2, 0);
                } else {
                    addView(imageView2, 0);
                }
            }
            if (drawable == null && !this.n) {
                this.f89c.setVisibility(8);
                return;
            }
            ImageView imageView3 = this.f89c;
            if (!z) {
                drawable = null;
            }
            imageView3.setImageDrawable(drawable);
            if (this.f89c.getVisibility() != 0) {
                this.f89c.setVisibility(0);
            }
        }
    }

    public void setTitle(CharSequence charSequence) {
        if (charSequence != null) {
            this.f91e.setText(charSequence);
            if (this.f91e.getVisibility() != 0) {
                this.f91e.setVisibility(0);
            }
        } else if (this.f91e.getVisibility() != 8) {
            this.f91e.setVisibility(8);
        }
    }
}