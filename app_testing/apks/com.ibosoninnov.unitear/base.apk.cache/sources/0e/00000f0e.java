package com.google.android.gms.common.server.response;

import android.os.Parcel;
import android.util.Log;
import androidx.annotation.RecentlyNonNull;
import androidx.annotation.RecentlyNullable;
import c.b.a.a.a;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.internal.Objects;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.internal.ShowFirstParty;
import com.google.android.gms.common.internal.safeparcel.AbstractSafeParcelable;
import com.google.android.gms.common.internal.safeparcel.SafeParcelWriter;
import com.google.android.gms.common.internal.safeparcel.SafeParcelable;
import com.google.android.gms.common.util.Base64Utils;
import com.google.android.gms.common.util.JsonUtils;
import com.google.android.gms.common.util.MapUtils;
import com.google.android.gms.common.util.VisibleForTesting;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
@ShowFirstParty
@KeepForSdk
/* loaded from: classes.dex */
public abstract class FastJsonResponse {

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    @VisibleForTesting
    @SafeParcelable.Class(creator = "FieldCreator")
    @ShowFirstParty
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class Field<I, O> extends AbstractSafeParcelable {
        public static final zaj CREATOR = new zaj();
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "getTypeIn", id = 2)
        public final int zaa;
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "isTypeInArray", id = 3)
        public final boolean zab;
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "getTypeOut", id = 4)
        public final int zac;
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "isTypeOutArray", id = 5)
        public final boolean zad;
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "getOutputFieldName", id = 6)
        public final String zae;
        @RecentlyNonNull
        @SafeParcelable.Field(getter = "getSafeParcelableFieldId", id = 7)
        public final int zaf;
        @RecentlyNullable
        public final Class<? extends FastJsonResponse> zag;
        @SafeParcelable.VersionField(getter = "getVersionCode", id = 1)
        private final int zah;
        @SafeParcelable.Field(getter = "getConcreteTypeName", id = 8)
        private final String zai;
        private zal zaj;
        @SafeParcelable.Field(getter = "getWrappedConverter", id = 9, type = "com.google.android.gms.common.server.converter.ConverterWrapper")
        private FieldConverter<I, O> zak;

        @SafeParcelable.Constructor
        public Field(@SafeParcelable.Param(id = 1) int i, @SafeParcelable.Param(id = 2) int i2, @SafeParcelable.Param(id = 3) boolean z, @SafeParcelable.Param(id = 4) int i3, @SafeParcelable.Param(id = 5) boolean z2, @SafeParcelable.Param(id = 6) String str, @SafeParcelable.Param(id = 7) int i4, @SafeParcelable.Param(id = 8) String str2, @SafeParcelable.Param(id = 9) com.google.android.gms.common.server.converter.zaa zaaVar) {
            this.zah = i;
            this.zaa = i2;
            this.zab = z;
            this.zac = i3;
            this.zad = z2;
            this.zae = str;
            this.zaf = i4;
            if (str2 == null) {
                this.zag = null;
                this.zai = null;
            } else {
                this.zag = SafeParcelResponse.class;
                this.zai = str2;
            }
            if (zaaVar == null) {
                this.zak = null;
            } else {
                this.zak = (FieldConverter<I, O>) zaaVar.zaa();
            }
        }

        @RecentlyNonNull
        @VisibleForTesting
        @KeepForSdk
        public static Field<byte[], byte[]> forBase64(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(8, false, 8, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<Boolean, Boolean> forBoolean(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(6, false, 6, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static <T extends FastJsonResponse> Field<T, T> forConcreteType(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull Class<T> cls) {
            return new Field<>(11, false, 11, false, str, i, cls, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static <T extends FastJsonResponse> Field<ArrayList<T>, ArrayList<T>> forConcreteTypeArray(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull Class<T> cls) {
            return new Field<>(11, true, 11, true, str, i, cls, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<Double, Double> forDouble(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(4, false, 4, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<Float, Float> forFloat(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(3, false, 3, false, str, i, null, null);
        }

        @RecentlyNonNull
        @VisibleForTesting
        @KeepForSdk
        public static Field<Integer, Integer> forInteger(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(0, false, 0, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<Long, Long> forLong(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(2, false, 2, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<String, String> forString(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(7, false, 7, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<HashMap<String, String>, HashMap<String, String>> forStringMap(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(10, false, 10, false, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field<ArrayList<String>, ArrayList<String>> forStrings(@RecentlyNonNull String str, @RecentlyNonNull int i) {
            return new Field<>(7, true, 7, true, str, i, null, null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public static Field withConverter(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull FieldConverter<?, ?> fieldConverter, @RecentlyNonNull boolean z) {
            return new Field(fieldConverter.zaa(), z, fieldConverter.zab(), false, str, i, null, fieldConverter);
        }

        private final String zae() {
            String str = this.zai;
            if (str == null) {
                return null;
            }
            return str;
        }

        private final com.google.android.gms.common.server.converter.zaa zaf() {
            FieldConverter<I, O> fieldConverter = this.zak;
            if (fieldConverter == null) {
                return null;
            }
            return com.google.android.gms.common.server.converter.zaa.zaa(fieldConverter);
        }

        @RecentlyNonNull
        @KeepForSdk
        public int getSafeParcelableFieldId() {
            return this.zaf;
        }

        @RecentlyNonNull
        public String toString() {
            Objects.ToStringHelper add = Objects.toStringHelper(this).add("versionCode", Integer.valueOf(this.zah)).add("typeIn", Integer.valueOf(this.zaa)).add("typeInArray", Boolean.valueOf(this.zab)).add("typeOut", Integer.valueOf(this.zac)).add("typeOutArray", Boolean.valueOf(this.zad)).add("outputFieldName", this.zae).add("safeParcelFieldId", Integer.valueOf(this.zaf)).add("concreteTypeName", zae());
            Class<? extends FastJsonResponse> cls = this.zag;
            if (cls != null) {
                add.add("concreteType.class", cls.getCanonicalName());
            }
            FieldConverter<I, O> fieldConverter = this.zak;
            if (fieldConverter != null) {
                add.add("converterName", fieldConverter.getClass().getCanonicalName());
            }
            return add.toString();
        }

        @Override // android.os.Parcelable
        public void writeToParcel(@RecentlyNonNull Parcel parcel, @RecentlyNonNull int i) {
            int beginObjectHeader = SafeParcelWriter.beginObjectHeader(parcel);
            SafeParcelWriter.writeInt(parcel, 1, this.zah);
            SafeParcelWriter.writeInt(parcel, 2, this.zaa);
            SafeParcelWriter.writeBoolean(parcel, 3, this.zab);
            SafeParcelWriter.writeInt(parcel, 4, this.zac);
            SafeParcelWriter.writeBoolean(parcel, 5, this.zad);
            SafeParcelWriter.writeString(parcel, 6, this.zae, false);
            SafeParcelWriter.writeInt(parcel, 7, getSafeParcelableFieldId());
            SafeParcelWriter.writeString(parcel, 8, zae(), false);
            SafeParcelWriter.writeParcelable(parcel, 9, zaf(), i, false);
            SafeParcelWriter.finishObjectHeader(parcel, beginObjectHeader);
        }

        @RecentlyNonNull
        public final Field<I, O> zaa() {
            return new Field<>(this.zah, this.zaa, this.zab, this.zac, this.zad, this.zae, this.zaf, this.zai, zaf());
        }

        @RecentlyNonNull
        public final boolean zab() {
            return this.zak != null;
        }

        @RecentlyNonNull
        public final FastJsonResponse zac() {
            Preconditions.checkNotNull(this.zag);
            Class<? extends FastJsonResponse> cls = this.zag;
            if (cls == SafeParcelResponse.class) {
                Preconditions.checkNotNull(this.zai);
                Preconditions.checkNotNull(this.zaj, "The field mapping dictionary must be set if the concrete type is a SafeParcelResponse object.");
                return new SafeParcelResponse(this.zaj, this.zai);
            }
            return cls.newInstance();
        }

        @RecentlyNonNull
        public final Map<String, Field<?, ?>> zad() {
            Preconditions.checkNotNull(this.zai);
            Preconditions.checkNotNull(this.zaj);
            return (Map) Preconditions.checkNotNull(this.zaj.zaa(this.zai));
        }

        @RecentlyNonNull
        public final I zab(@RecentlyNonNull O o) {
            Preconditions.checkNotNull(this.zak);
            return this.zak.zaa(o);
        }

        public final void zaa(zal zalVar) {
            this.zaj = zalVar;
        }

        @RecentlyNonNull
        public final O zaa(I i) {
            Preconditions.checkNotNull(this.zak);
            return (O) Preconditions.checkNotNull(this.zak.zab(i));
        }

        private Field(int i, boolean z, int i2, boolean z2, String str, int i3, Class<? extends FastJsonResponse> cls, FieldConverter<I, O> fieldConverter) {
            this.zah = 1;
            this.zaa = i;
            this.zab = z;
            this.zac = i2;
            this.zad = z2;
            this.zae = str;
            this.zaf = i3;
            this.zag = cls;
            if (cls == null) {
                this.zai = null;
            } else {
                this.zai = cls.getCanonicalName();
            }
            this.zak = fieldConverter;
        }
    }

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    @ShowFirstParty
    /* loaded from: classes.dex */
    public interface FieldConverter<I, O> {
        @RecentlyNonNull
        int zaa();

        @RecentlyNonNull
        I zaa(@RecentlyNonNull O o);

        @RecentlyNonNull
        int zab();

        @RecentlyNullable
        O zab(@RecentlyNonNull I i);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r2v0, resolved type: java.lang.Object */
    /* JADX WARN: Multi-variable type inference failed */
    @RecentlyNonNull
    public static <O, I> I zaa(@RecentlyNonNull Field<I, O> field, Object obj) {
        return ((Field) field).zak != null ? field.zab(obj) : obj;
    }

    private final <I, O> void zab(Field<I, O> field, I i) {
        String str = field.zae;
        O zaa = field.zaa((Field<I, O>) i);
        switch (field.zac) {
            case 0:
                if (zaa != null) {
                    setIntegerInternal(field, str, ((Integer) zaa).intValue());
                    return;
                } else {
                    zaa(str);
                    return;
                }
            case 1:
                zaa((Field<?, ?>) field, str, (BigInteger) zaa);
                return;
            case 2:
                if (zaa != null) {
                    setLongInternal(field, str, ((Long) zaa).longValue());
                    return;
                } else {
                    zaa(str);
                    return;
                }
            case 3:
            default:
                throw new IllegalStateException(a.g(44, "Unsupported type for conversion: ", field.zac));
            case 4:
                if (zaa != null) {
                    zaa((Field<?, ?>) field, str, ((Double) zaa).doubleValue());
                    return;
                } else {
                    zaa(str);
                    return;
                }
            case 5:
                zaa((Field<?, ?>) field, str, (BigDecimal) zaa);
                return;
            case 6:
                if (zaa != null) {
                    setBooleanInternal(field, str, ((Boolean) zaa).booleanValue());
                    return;
                } else {
                    zaa(str);
                    return;
                }
            case 7:
                setStringInternal(field, str, (String) zaa);
                return;
            case 8:
            case 9:
                if (zaa != null) {
                    setDecodedBytesInternal(field, str, (byte[]) zaa);
                    return;
                } else {
                    zaa(str);
                    return;
                }
        }
    }

    @KeepForSdk
    public <T extends FastJsonResponse> void addConcreteTypeArrayInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<T> arrayList) {
        throw new UnsupportedOperationException("Concrete type array not supported");
    }

    @KeepForSdk
    public <T extends FastJsonResponse> void addConcreteTypeInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull T t) {
        throw new UnsupportedOperationException("Concrete type not supported");
    }

    @RecentlyNonNull
    @KeepForSdk
    public abstract Map<String, Field<?, ?>> getFieldMappings();

    @RecentlyNullable
    @KeepForSdk
    public Object getFieldValue(@RecentlyNonNull Field field) {
        String str = field.zae;
        if (field.zag != null) {
            Preconditions.checkState(getValueObject(str) == null, "Concrete field shouldn't be value object: %s", field.zae);
            try {
                char upperCase = Character.toUpperCase(str.charAt(0));
                String substring = str.substring(1);
                StringBuilder sb = new StringBuilder(String.valueOf(substring).length() + 4);
                sb.append("get");
                sb.append(upperCase);
                sb.append(substring);
                return getClass().getMethod(sb.toString(), new Class[0]).invoke(this, new Object[0]);
            } catch (Exception e2) {
                throw new RuntimeException(e2);
            }
        }
        return getValueObject(str);
    }

    @RecentlyNullable
    @KeepForSdk
    public abstract Object getValueObject(@RecentlyNonNull String str);

    @RecentlyNonNull
    @KeepForSdk
    public boolean isFieldSet(@RecentlyNonNull Field field) {
        if (field.zac == 11) {
            if (field.zad) {
                throw new UnsupportedOperationException("Concrete type arrays not supported");
            }
            throw new UnsupportedOperationException("Concrete types not supported");
        }
        return isPrimitiveFieldSet(field.zae);
    }

    @RecentlyNonNull
    @KeepForSdk
    public abstract boolean isPrimitiveFieldSet(@RecentlyNonNull String str);

    @KeepForSdk
    public void setBooleanInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull boolean z) {
        throw new UnsupportedOperationException("Boolean not supported");
    }

    @KeepForSdk
    public void setDecodedBytesInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, byte[] bArr) {
        throw new UnsupportedOperationException("byte[] not supported");
    }

    @KeepForSdk
    public void setIntegerInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull int i) {
        throw new UnsupportedOperationException("Integer not supported");
    }

    @KeepForSdk
    public void setLongInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull long j) {
        throw new UnsupportedOperationException("Long not supported");
    }

    @KeepForSdk
    public void setStringInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, String str2) {
        throw new UnsupportedOperationException("String not supported");
    }

    @KeepForSdk
    public void setStringMapInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, Map<String, String> map) {
        throw new UnsupportedOperationException("String map not supported");
    }

    @KeepForSdk
    public void setStringsInternal(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<String> arrayList) {
        throw new UnsupportedOperationException("String list not supported");
    }

    @RecentlyNonNull
    @KeepForSdk
    public String toString() {
        Map<String, Field<?, ?>> fieldMappings = getFieldMappings();
        StringBuilder sb = new StringBuilder(100);
        for (String str : fieldMappings.keySet()) {
            Field<?, ?> field = fieldMappings.get(str);
            if (isFieldSet(field)) {
                Object zaa = zaa(field, getFieldValue(field));
                if (sb.length() == 0) {
                    sb.append("{");
                } else {
                    sb.append(",");
                }
                sb.append("\"");
                sb.append(str);
                sb.append("\":");
                if (zaa == null) {
                    sb.append("null");
                } else {
                    switch (field.zac) {
                        case 8:
                            sb.append("\"");
                            sb.append(Base64Utils.encode((byte[]) zaa));
                            sb.append("\"");
                            continue;
                        case 9:
                            sb.append("\"");
                            sb.append(Base64Utils.encodeUrlSafe((byte[]) zaa));
                            sb.append("\"");
                            continue;
                        case 10:
                            MapUtils.writeStringMapToJson(sb, (HashMap) zaa);
                            continue;
                        default:
                            if (field.zab) {
                                ArrayList arrayList = (ArrayList) zaa;
                                sb.append("[");
                                int size = arrayList.size();
                                for (int i = 0; i < size; i++) {
                                    if (i > 0) {
                                        sb.append(",");
                                    }
                                    Object obj = arrayList.get(i);
                                    if (obj != null) {
                                        zaa(sb, field, obj);
                                    }
                                }
                                sb.append("]");
                                break;
                            } else {
                                zaa(sb, field, zaa);
                                continue;
                            }
                    }
                }
            }
        }
        if (sb.length() > 0) {
            sb.append("}");
        } else {
            sb.append("{}");
        }
        return sb.toString();
    }

    public final <O> void zac(@RecentlyNonNull Field<ArrayList<Long>, O> field, ArrayList<Long> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<Long>, O>, O>) field, (Field<ArrayList<Long>, O>) arrayList);
        } else {
            zac(field, field.zae, arrayList);
        }
    }

    public final <O> void zad(@RecentlyNonNull Field<ArrayList<Float>, O> field, ArrayList<Float> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<Float>, O>, O>) field, (Field<ArrayList<Float>, O>) arrayList);
        } else {
            zad(field, field.zae, arrayList);
        }
    }

    public final <O> void zae(@RecentlyNonNull Field<ArrayList<Double>, O> field, ArrayList<Double> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<Double>, O>, O>) field, (Field<ArrayList<Double>, O>) arrayList);
        } else {
            zae(field, field.zae, arrayList);
        }
    }

    public final <O> void zaf(@RecentlyNonNull Field<ArrayList<BigDecimal>, O> field, ArrayList<BigDecimal> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<BigDecimal>, O>, O>) field, (Field<ArrayList<BigDecimal>, O>) arrayList);
        } else {
            zaf(field, field.zae, arrayList);
        }
    }

    public final <O> void zag(@RecentlyNonNull Field<ArrayList<Boolean>, O> field, ArrayList<Boolean> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<Boolean>, O>, O>) field, (Field<ArrayList<Boolean>, O>) arrayList);
        } else {
            zag(field, field.zae, arrayList);
        }
    }

    public final <O> void zah(@RecentlyNonNull Field<ArrayList<String>, O> field, ArrayList<String> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<String>, O>, O>) field, (Field<ArrayList<String>, O>) arrayList);
        } else {
            setStringsInternal(field, field.zae, arrayList);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Integer, O> field, @RecentlyNonNull int i) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Integer, O>, O>) field, (Field<Integer, O>) Integer.valueOf(i));
        } else {
            setIntegerInternal(field, field.zae, i);
        }
    }

    public void zac(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<Long> arrayList) {
        throw new UnsupportedOperationException("Long list not supported");
    }

    public void zad(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<Float> arrayList) {
        throw new UnsupportedOperationException("Float list not supported");
    }

    public void zae(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<Double> arrayList) {
        throw new UnsupportedOperationException("Double list not supported");
    }

    public void zaf(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<BigDecimal> arrayList) {
        throw new UnsupportedOperationException("BigDecimal list not supported");
    }

    public void zag(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<Boolean> arrayList) {
        throw new UnsupportedOperationException("Boolean list not supported");
    }

    public final <O> void zaa(@RecentlyNonNull Field<ArrayList<Integer>, O> field, ArrayList<Integer> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<Integer>, O>, O>) field, (Field<ArrayList<Integer>, O>) arrayList);
        } else {
            zaa(field, field.zae, arrayList);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<BigInteger, O> field, BigInteger bigInteger) {
        if (((Field) field).zak != null) {
            zab((Field<Field<BigInteger, O>, O>) field, (Field<BigInteger, O>) bigInteger);
        } else {
            zaa(field, field.zae, bigInteger);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Long, O> field, @RecentlyNonNull long j) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Long, O>, O>) field, (Field<Long, O>) Long.valueOf(j));
        } else {
            setLongInternal(field, field.zae, j);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Float, O> field, @RecentlyNonNull float f2) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Float, O>, O>) field, (Field<Float, O>) Float.valueOf(f2));
        } else {
            zaa((Field<?, ?>) field, field.zae, f2);
        }
    }

    public final <O> void zab(@RecentlyNonNull Field<ArrayList<BigInteger>, O> field, ArrayList<BigInteger> arrayList) {
        if (((Field) field).zak != null) {
            zab((Field<Field<ArrayList<BigInteger>, O>, O>) field, (Field<ArrayList<BigInteger>, O>) arrayList);
        } else {
            zab(field, field.zae, arrayList);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Double, O> field, @RecentlyNonNull double d2) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Double, O>, O>) field, (Field<Double, O>) Double.valueOf(d2));
        } else {
            zaa(field, field.zae, d2);
        }
    }

    public void zab(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<BigInteger> arrayList) {
        throw new UnsupportedOperationException("BigInteger list not supported");
    }

    public final <O> void zaa(@RecentlyNonNull Field<BigDecimal, O> field, BigDecimal bigDecimal) {
        if (((Field) field).zak != null) {
            zab((Field<Field<BigDecimal, O>, O>) field, (Field<BigDecimal, O>) bigDecimal);
        } else {
            zaa(field, field.zae, bigDecimal);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Boolean, O> field, @RecentlyNonNull boolean z) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Boolean, O>, O>) field, (Field<Boolean, O>) Boolean.valueOf(z));
        } else {
            setBooleanInternal(field, field.zae, z);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<String, O> field, String str) {
        if (((Field) field).zak != null) {
            zab((Field<Field<String, O>, O>) field, (Field<String, O>) str);
        } else {
            setStringInternal(field, field.zae, str);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<byte[], O> field, byte[] bArr) {
        if (((Field) field).zak != null) {
            zab((Field<Field<byte[], O>, O>) field, (Field<byte[], O>) bArr);
        } else {
            setDecodedBytesInternal(field, field.zae, bArr);
        }
    }

    public final <O> void zaa(@RecentlyNonNull Field<Map<String, String>, O> field, Map<String, String> map) {
        if (((Field) field).zak != null) {
            zab((Field<Field<Map<String, String>, O>, O>) field, (Field<Map<String, String>, O>) map);
        } else {
            setStringMapInternal(field, field.zae, map);
        }
    }

    public void zaa(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, ArrayList<Integer> arrayList) {
        throw new UnsupportedOperationException("Integer list not supported");
    }

    public void zaa(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, BigInteger bigInteger) {
        throw new UnsupportedOperationException("BigInteger not supported");
    }

    public void zaa(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull float f2) {
        throw new UnsupportedOperationException("Float not supported");
    }

    public void zaa(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, @RecentlyNonNull double d2) {
        throw new UnsupportedOperationException("Double not supported");
    }

    public void zaa(@RecentlyNonNull Field<?, ?> field, @RecentlyNonNull String str, BigDecimal bigDecimal) {
        throw new UnsupportedOperationException("BigDecimal not supported");
    }

    private static <O> void zaa(String str) {
        if (Log.isLoggable("FastJsonResponse", 6)) {
            StringBuilder sb = new StringBuilder(String.valueOf(str).length() + 58);
            sb.append("Output field (");
            sb.append(str);
            sb.append(") has a null value, but expected a primitive");
            Log.e("FastJsonResponse", sb.toString());
        }
    }

    private static void zaa(StringBuilder sb, Field field, Object obj) {
        int i = field.zaa;
        if (i == 11) {
            Class<? extends FastJsonResponse> cls = field.zag;
            Preconditions.checkNotNull(cls);
            sb.append(cls.cast(obj).toString());
        } else if (i == 7) {
            sb.append("\"");
            sb.append(JsonUtils.escapeString((String) obj));
            sb.append("\"");
        } else {
            sb.append(obj);
        }
    }
}