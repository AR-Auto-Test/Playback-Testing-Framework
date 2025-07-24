package b.z;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Picture;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import b.z.j;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;

/* compiled from: Visibility.java */
/* loaded from: classes.dex */
public abstract class z extends j {
    public static final int MODE_IN = 1;
    public static final int MODE_OUT = 2;
    private static final String PROPNAME_SCREEN_LOCATION = "android:visibility:screenLocation";
    private int mMode;
    public static final String PROPNAME_VISIBILITY = "android:visibility:visibility";
    private static final String PROPNAME_PARENT = "android:visibility:parent";
    private static final String[] sTransitionProperties = {PROPNAME_VISIBILITY, PROPNAME_PARENT};

    /* compiled from: Visibility.java */
    /* loaded from: classes.dex */
    public class a extends k {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ ViewGroup f2930a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ View f2931b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ View f2932c;

        public a(ViewGroup viewGroup, View view, View view2) {
            this.f2930a = viewGroup;
            this.f2931b = view;
            this.f2932c = view2;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            this.f2932c.setTag(R.id.save_overlay_view, null);
            this.f2930a.getOverlay().remove(this.f2931b);
            jVar.removeListener(this);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionPause(j jVar) {
            this.f2930a.getOverlay().remove(this.f2931b);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionResume(j jVar) {
            if (this.f2931b.getParent() == null) {
                this.f2930a.getOverlay().add(this.f2931b);
            } else {
                z.this.cancel();
            }
        }
    }

    /* compiled from: Visibility.java */
    /* loaded from: classes.dex */
    public static class b extends AnimatorListenerAdapter implements j.f {

        /* renamed from: a  reason: collision with root package name */
        public final View f2934a;

        /* renamed from: b  reason: collision with root package name */
        public final int f2935b;

        /* renamed from: c  reason: collision with root package name */
        public final ViewGroup f2936c;

        /* renamed from: d  reason: collision with root package name */
        public final boolean f2937d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f2938e;

        /* renamed from: f  reason: collision with root package name */
        public boolean f2939f = false;

        public b(View view, int i, boolean z) {
            this.f2934a = view;
            this.f2935b = i;
            this.f2936c = (ViewGroup) view.getParent();
            this.f2937d = z;
            b(true);
        }

        public final void a() {
            if (!this.f2939f) {
                s.f2921a.f(this.f2934a, this.f2935b);
                ViewGroup viewGroup = this.f2936c;
                if (viewGroup != null) {
                    viewGroup.invalidate();
                }
            }
            b(false);
        }

        public final void b(boolean z) {
            ViewGroup viewGroup;
            if (!this.f2937d || this.f2938e == z || (viewGroup = this.f2936c) == null) {
                return;
            }
            this.f2938e = z;
            r.a(viewGroup, z);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationCancel(Animator animator) {
            this.f2939f = true;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            a();
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorPauseListener
        public void onAnimationPause(Animator animator) {
            if (this.f2939f) {
                return;
            }
            s.f2921a.f(this.f2934a, this.f2935b);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationRepeat(Animator animator) {
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorPauseListener
        public void onAnimationResume(Animator animator) {
            if (this.f2939f) {
                return;
            }
            s.f2921a.f(this.f2934a, 0);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationStart(Animator animator) {
        }

        @Override // b.z.j.f
        public void onTransitionCancel(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            a();
            jVar.removeListener(this);
        }

        @Override // b.z.j.f
        public void onTransitionPause(j jVar) {
            b(false);
        }

        @Override // b.z.j.f
        public void onTransitionResume(j jVar) {
            b(true);
        }

        @Override // b.z.j.f
        public void onTransitionStart(j jVar) {
        }
    }

    /* compiled from: Visibility.java */
    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public boolean f2940a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f2941b;

        /* renamed from: c  reason: collision with root package name */
        public int f2942c;

        /* renamed from: d  reason: collision with root package name */
        public int f2943d;

        /* renamed from: e  reason: collision with root package name */
        public ViewGroup f2944e;

        /* renamed from: f  reason: collision with root package name */
        public ViewGroup f2945f;
    }

    public z() {
        this.mMode = 3;
    }

    private void captureValues(p pVar) {
        pVar.f2913a.put(PROPNAME_VISIBILITY, Integer.valueOf(pVar.f2914b.getVisibility()));
        pVar.f2913a.put(PROPNAME_PARENT, pVar.f2914b.getParent());
        int[] iArr = new int[2];
        pVar.f2914b.getLocationOnScreen(iArr);
        pVar.f2913a.put(PROPNAME_SCREEN_LOCATION, iArr);
    }

    private c getVisibilityChangeInfo(p pVar, p pVar2) {
        c cVar = new c();
        cVar.f2940a = false;
        cVar.f2941b = false;
        if (pVar != null && pVar.f2913a.containsKey(PROPNAME_VISIBILITY)) {
            cVar.f2942c = ((Integer) pVar.f2913a.get(PROPNAME_VISIBILITY)).intValue();
            cVar.f2944e = (ViewGroup) pVar.f2913a.get(PROPNAME_PARENT);
        } else {
            cVar.f2942c = -1;
            cVar.f2944e = null;
        }
        if (pVar2 != null && pVar2.f2913a.containsKey(PROPNAME_VISIBILITY)) {
            cVar.f2943d = ((Integer) pVar2.f2913a.get(PROPNAME_VISIBILITY)).intValue();
            cVar.f2945f = (ViewGroup) pVar2.f2913a.get(PROPNAME_PARENT);
        } else {
            cVar.f2943d = -1;
            cVar.f2945f = null;
        }
        if (pVar != null && pVar2 != null) {
            int i = cVar.f2942c;
            int i2 = cVar.f2943d;
            if (i == i2 && cVar.f2944e == cVar.f2945f) {
                return cVar;
            }
            if (i != i2) {
                if (i == 0) {
                    cVar.f2941b = false;
                    cVar.f2940a = true;
                } else if (i2 == 0) {
                    cVar.f2941b = true;
                    cVar.f2940a = true;
                }
            } else if (cVar.f2945f == null) {
                cVar.f2941b = false;
                cVar.f2940a = true;
            } else if (cVar.f2944e == null) {
                cVar.f2941b = true;
                cVar.f2940a = true;
            }
        } else if (pVar == null && cVar.f2943d == 0) {
            cVar.f2941b = true;
            cVar.f2940a = true;
        } else if (pVar2 == null && cVar.f2942c == 0) {
            cVar.f2941b = false;
            cVar.f2940a = true;
        }
        return cVar;
    }

    @Override // b.z.j
    public void captureEndValues(p pVar) {
        captureValues(pVar);
    }

    @Override // b.z.j
    public void captureStartValues(p pVar) {
        captureValues(pVar);
    }

    @Override // b.z.j
    public Animator createAnimator(ViewGroup viewGroup, p pVar, p pVar2) {
        c visibilityChangeInfo = getVisibilityChangeInfo(pVar, pVar2);
        if (visibilityChangeInfo.f2940a) {
            if (visibilityChangeInfo.f2944e == null && visibilityChangeInfo.f2945f == null) {
                return null;
            }
            if (visibilityChangeInfo.f2941b) {
                return onAppear(viewGroup, pVar, visibilityChangeInfo.f2942c, pVar2, visibilityChangeInfo.f2943d);
            }
            return onDisappear(viewGroup, pVar, visibilityChangeInfo.f2942c, pVar2, visibilityChangeInfo.f2943d);
        }
        return null;
    }

    public int getMode() {
        return this.mMode;
    }

    @Override // b.z.j
    public String[] getTransitionProperties() {
        return sTransitionProperties;
    }

    @Override // b.z.j
    public boolean isTransitionRequired(p pVar, p pVar2) {
        if (pVar == null && pVar2 == null) {
            return false;
        }
        if (pVar == null || pVar2 == null || pVar2.f2913a.containsKey(PROPNAME_VISIBILITY) == pVar.f2913a.containsKey(PROPNAME_VISIBILITY)) {
            c visibilityChangeInfo = getVisibilityChangeInfo(pVar, pVar2);
            if (visibilityChangeInfo.f2940a) {
                return visibilityChangeInfo.f2942c == 0 || visibilityChangeInfo.f2943d == 0;
            }
            return false;
        }
        return false;
    }

    public boolean isVisible(p pVar) {
        if (pVar == null) {
            return false;
        }
        return ((Integer) pVar.f2913a.get(PROPNAME_VISIBILITY)).intValue() == 0 && ((View) pVar.f2913a.get(PROPNAME_PARENT)) != null;
    }

    public abstract Animator onAppear(ViewGroup viewGroup, View view, p pVar, p pVar2);

    public Animator onAppear(ViewGroup viewGroup, p pVar, int i, p pVar2, int i2) {
        if ((this.mMode & 1) != 1 || pVar2 == null) {
            return null;
        }
        if (pVar == null) {
            View view = (View) pVar2.f2914b.getParent();
            if (getVisibilityChangeInfo(getMatchedTransitionValues(view, false), getTransitionValues(view, false)).f2940a) {
                return null;
            }
        }
        return onAppear(viewGroup, pVar2.f2914b, pVar, pVar2);
    }

    public abstract Animator onDisappear(ViewGroup viewGroup, View view, p pVar, p pVar2);

    /* JADX WARN: Code restructure failed: missing block: B:70:0x01b0, code lost:
        if (r0.mCanRemoveViews != false) goto L42;
     */
    /* JADX WARN: Removed duplicated region for block: B:27:0x0049  */
    /* JADX WARN: Removed duplicated region for block: B:61:0x017c  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public Animator onDisappear(ViewGroup viewGroup, p pVar, int i, p pVar2, int i2) {
        boolean z;
        View view;
        View view2;
        boolean z2;
        boolean z3;
        boolean z4;
        int i3;
        ViewGroup viewGroup2;
        int round;
        Bitmap bitmap;
        z zVar = this;
        if ((zVar.mMode & 2) == 2 && pVar != null) {
            View view3 = pVar.f2914b;
            View view4 = pVar2 != null ? pVar2.f2914b : null;
            View view5 = (View) view3.getTag(R.id.save_overlay_view);
            boolean z5 = true;
            if (view5 != null) {
                view2 = null;
            } else if (view4 == null || view4.getParent() == null) {
                if (view4 != null) {
                    view5 = view4;
                    z = false;
                    view4 = null;
                    if (z) {
                        if (view3.getParent() == null) {
                            view = view4;
                        } else if (view3.getParent() instanceof View) {
                            View view6 = (View) view3.getParent();
                            if (!zVar.getVisibilityChangeInfo(zVar.getTransitionValues(view6, true), zVar.getMatchedTransitionValues(view6, true)).f2940a) {
                                boolean z6 = o.f2910a;
                                Matrix matrix = new Matrix();
                                matrix.setTranslate(-view6.getScrollX(), -view6.getScrollY());
                                y yVar = s.f2921a;
                                yVar.g(view3, matrix);
                                yVar.h(viewGroup, matrix);
                                RectF rectF = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, view3.getWidth(), view3.getHeight());
                                matrix.mapRect(rectF);
                                int round2 = Math.round(rectF.left);
                                int round3 = Math.round(rectF.top);
                                int round4 = Math.round(rectF.right);
                                int round5 = Math.round(rectF.bottom);
                                ImageView imageView = new ImageView(view3.getContext());
                                imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                                if (o.f2910a) {
                                    z2 = !view3.isAttachedToWindow();
                                    if (viewGroup != null) {
                                        z3 = viewGroup.isAttachedToWindow();
                                        z4 = o.f2911b;
                                        if (z4 || !z2) {
                                            view = view4;
                                            i3 = 0;
                                            viewGroup2 = null;
                                        } else if (z3) {
                                            viewGroup2 = (ViewGroup) view3.getParent();
                                            int indexOfChild = viewGroup2.indexOfChild(view3);
                                            view = view4;
                                            viewGroup.getOverlay().add(view3);
                                            i3 = indexOfChild;
                                        } else {
                                            view = view4;
                                            bitmap = null;
                                            if (bitmap != null) {
                                                imageView.setImageBitmap(bitmap);
                                            }
                                            imageView.measure(View.MeasureSpec.makeMeasureSpec(round4 - round2, 1073741824), View.MeasureSpec.makeMeasureSpec(round5 - round3, 1073741824));
                                            imageView.layout(round2, round3, round4, round5);
                                            zVar = this;
                                            view5 = imageView;
                                        }
                                        round = Math.round(rectF.width());
                                        int round6 = Math.round(rectF.height());
                                        if (round > 0 || round6 <= 0) {
                                            bitmap = null;
                                        } else {
                                            float min = Math.min(1.0f, 1048576.0f / (round * round6));
                                            int round7 = Math.round(round * min);
                                            int round8 = Math.round(round6 * min);
                                            matrix.postTranslate(-rectF.left, -rectF.top);
                                            matrix.postScale(min, min);
                                            if (o.f2912c) {
                                                Picture picture = new Picture();
                                                Canvas beginRecording = picture.beginRecording(round7, round8);
                                                beginRecording.concat(matrix);
                                                view3.draw(beginRecording);
                                                picture.endRecording();
                                                bitmap = Bitmap.createBitmap(picture);
                                            } else {
                                                bitmap = Bitmap.createBitmap(round7, round8, Bitmap.Config.ARGB_8888);
                                                Canvas canvas = new Canvas(bitmap);
                                                canvas.concat(matrix);
                                                view3.draw(canvas);
                                            }
                                        }
                                        if (z4 && z2) {
                                            viewGroup.getOverlay().remove(view3);
                                            viewGroup2.addView(view3, i3);
                                        }
                                        if (bitmap != null) {
                                        }
                                        imageView.measure(View.MeasureSpec.makeMeasureSpec(round4 - round2, 1073741824), View.MeasureSpec.makeMeasureSpec(round5 - round3, 1073741824));
                                        imageView.layout(round2, round3, round4, round5);
                                        zVar = this;
                                        view5 = imageView;
                                    }
                                } else {
                                    z2 = false;
                                }
                                z3 = false;
                                z4 = o.f2911b;
                                if (z4) {
                                }
                                view = view4;
                                i3 = 0;
                                viewGroup2 = null;
                                round = Math.round(rectF.width());
                                int round62 = Math.round(rectF.height());
                                if (round > 0) {
                                }
                                bitmap = null;
                                if (z4) {
                                    viewGroup.getOverlay().remove(view3);
                                    viewGroup2.addView(view3, i3);
                                }
                                if (bitmap != null) {
                                }
                                imageView.measure(View.MeasureSpec.makeMeasureSpec(round4 - round2, 1073741824), View.MeasureSpec.makeMeasureSpec(round5 - round3, 1073741824));
                                imageView.layout(round2, round3, round4, round5);
                                zVar = this;
                                view5 = imageView;
                            } else {
                                view = view4;
                                int id = view6.getId();
                                if (view6.getParent() != null || id == -1 || viewGroup.findViewById(id) == null) {
                                    zVar = this;
                                } else {
                                    zVar = this;
                                }
                            }
                            view2 = view;
                            z5 = false;
                        }
                        view5 = view3;
                        view2 = view;
                        z5 = false;
                    }
                    view = view4;
                    view2 = view;
                    z5 = false;
                }
                view4 = null;
                view5 = null;
                z = true;
                if (z) {
                }
                view = view4;
                view2 = view;
                z5 = false;
            } else {
                if (i2 == 4 || view3 == view4) {
                    view5 = null;
                    z = false;
                    if (z) {
                    }
                    view = view4;
                    view2 = view;
                    z5 = false;
                }
                view4 = null;
                view5 = null;
                z = true;
                if (z) {
                }
                view = view4;
                view2 = view;
                z5 = false;
            }
            if (view5 == null) {
                if (view2 != null) {
                    int visibility = view2.getVisibility();
                    y yVar2 = s.f2921a;
                    yVar2.f(view2, 0);
                    Animator onDisappear = zVar.onDisappear(viewGroup, view2, pVar, pVar2);
                    if (onDisappear != null) {
                        b bVar = new b(view2, i2, true);
                        onDisappear.addListener(bVar);
                        onDisappear.addPauseListener(bVar);
                        zVar.addListener(bVar);
                    } else {
                        yVar2.f(view2, visibility);
                    }
                    return onDisappear;
                }
                return null;
            }
            if (!z5) {
                int[] iArr = (int[]) pVar.f2913a.get(PROPNAME_SCREEN_LOCATION);
                int i4 = iArr[0];
                int i5 = iArr[1];
                int[] iArr2 = new int[2];
                viewGroup.getLocationOnScreen(iArr2);
                view5.offsetLeftAndRight((i4 - iArr2[0]) - view5.getLeft());
                view5.offsetTopAndBottom((i5 - iArr2[1]) - view5.getTop());
                viewGroup.getOverlay().add(view5);
            }
            Animator onDisappear2 = zVar.onDisappear(viewGroup, view5, pVar, pVar2);
            if (!z5) {
                if (onDisappear2 == null) {
                    viewGroup.getOverlay().remove(view5);
                } else {
                    view3.setTag(R.id.save_overlay_view, view5);
                    zVar.addListener(new a(viewGroup, view5, view3));
                }
            }
            return onDisappear2;
        }
        return null;
    }

    public void setMode(int i) {
        if ((i & (-4)) == 0) {
            this.mMode = i;
            return;
        }
        throw new IllegalArgumentException("Only MODE_IN and MODE_OUT flags are allowed");
    }

    @SuppressLint({"RestrictedApi"})
    public z(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mMode = 3;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2887b);
        int b2 = b.j.c.b.f.b(obtainStyledAttributes, (XmlResourceParser) attributeSet, "transitionVisibilityMode", 0, 0);
        obtainStyledAttributes.recycle();
        if (b2 != 0) {
            setMode(b2);
        }
    }
}