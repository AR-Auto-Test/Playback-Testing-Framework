package com.google.protobuf;

import c.b.a.a.a;
import com.google.protobuf.Descriptors;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.GeneratedMessageV3;
import com.google.protobuf.WireFormat;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/* loaded from: classes2.dex */
public final class ExtensionSchemaFull extends ExtensionSchema<Descriptors.FieldDescriptor> {
    private static final long EXTENSION_FIELD_OFFSET = getExtensionsFieldOffset();

    /* renamed from: com.google.protobuf.ExtensionSchemaFull$1  reason: invalid class name */
    /* loaded from: classes2.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$protobuf$WireFormat$FieldType;

        static {
            WireFormat.FieldType.values();
            int[] iArr = new int[18];
            $SwitchMap$com$google$protobuf$WireFormat$FieldType = iArr;
            try {
                iArr[WireFormat.FieldType.DOUBLE.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.FLOAT.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.INT64.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.UINT64.ordinal()] = 4;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.INT32.ordinal()] = 5;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.FIXED64.ordinal()] = 6;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.FIXED32.ordinal()] = 7;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.BOOL.ordinal()] = 8;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.UINT32.ordinal()] = 9;
            } catch (NoSuchFieldError unused9) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.SFIXED32.ordinal()] = 10;
            } catch (NoSuchFieldError unused10) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.SFIXED64.ordinal()] = 11;
            } catch (NoSuchFieldError unused11) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.SINT32.ordinal()] = 12;
            } catch (NoSuchFieldError unused12) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.SINT64.ordinal()] = 13;
            } catch (NoSuchFieldError unused13) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.ENUM.ordinal()] = 14;
            } catch (NoSuchFieldError unused14) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.BYTES.ordinal()] = 15;
            } catch (NoSuchFieldError unused15) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.STRING.ordinal()] = 16;
            } catch (NoSuchFieldError unused16) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.GROUP.ordinal()] = 17;
            } catch (NoSuchFieldError unused17) {
            }
            try {
                $SwitchMap$com$google$protobuf$WireFormat$FieldType[WireFormat.FieldType.MESSAGE.ordinal()] = 18;
            } catch (NoSuchFieldError unused18) {
            }
        }
    }

    private static <T> long getExtensionsFieldOffset() {
        return UnsafeUtil.objectFieldOffset(GeneratedMessageV3.ExtendableMessage.class.getDeclaredField("extensions"));
    }

    @Override // com.google.protobuf.ExtensionSchema
    public int extensionNumber(Map.Entry<?, ?> entry) {
        return ((Descriptors.FieldDescriptor) entry.getKey()).getNumber();
    }

    @Override // com.google.protobuf.ExtensionSchema
    public Object findExtensionByNumber(ExtensionRegistryLite extensionRegistryLite, MessageLite messageLite, int i) {
        return ((ExtensionRegistry) extensionRegistryLite).findExtensionByNumber(((Message) messageLite).getDescriptorForType(), i);
    }

    @Override // com.google.protobuf.ExtensionSchema
    public FieldSet<Descriptors.FieldDescriptor> getExtensions(Object obj) {
        return (FieldSet) UnsafeUtil.getObject(obj, EXTENSION_FIELD_OFFSET);
    }

    @Override // com.google.protobuf.ExtensionSchema
    public FieldSet<Descriptors.FieldDescriptor> getMutableExtensions(Object obj) {
        FieldSet<Descriptors.FieldDescriptor> extensions = getExtensions(obj);
        if (extensions.isImmutable()) {
            FieldSet<Descriptors.FieldDescriptor> m138clone = extensions.m138clone();
            setExtensions(obj, m138clone);
            return m138clone;
        }
        return extensions;
    }

    @Override // com.google.protobuf.ExtensionSchema
    public boolean hasExtensions(MessageLite messageLite) {
        return messageLite instanceof GeneratedMessageV3.ExtendableMessage;
    }

    @Override // com.google.protobuf.ExtensionSchema
    public void makeImmutable(Object obj) {
        getExtensions(obj).makeImmutable();
    }

    @Override // com.google.protobuf.ExtensionSchema
    public <UT, UB> UB parseExtension(Reader reader, Object obj, ExtensionRegistryLite extensionRegistryLite, FieldSet<Descriptors.FieldDescriptor> fieldSet, UB ub, UnknownFieldSchema<UT, UB> unknownFieldSchema) {
        Object field;
        List<Long> arrayList;
        ArrayList arrayList2;
        ExtensionRegistry.ExtensionInfo extensionInfo = (ExtensionRegistry.ExtensionInfo) obj;
        int number = extensionInfo.descriptor.getNumber();
        if (extensionInfo.descriptor.isRepeated() && extensionInfo.descriptor.isPacked()) {
            switch (extensionInfo.descriptor.getLiteType().ordinal()) {
                case 0:
                    arrayList = new ArrayList<>();
                    reader.readDoubleList(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 1:
                    arrayList = new ArrayList<>();
                    reader.readFloatList(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 2:
                    arrayList = new ArrayList<>();
                    reader.readInt64List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 3:
                    arrayList = new ArrayList<>();
                    reader.readUInt64List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 4:
                    arrayList = new ArrayList<>();
                    reader.readInt32List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 5:
                    arrayList = new ArrayList<>();
                    reader.readFixed64List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 6:
                    arrayList = new ArrayList<>();
                    reader.readFixed32List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 7:
                    arrayList = new ArrayList<>();
                    reader.readBoolList(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 8:
                case 9:
                case 10:
                case 11:
                default:
                    StringBuilder x = a.x("Type cannot be packed: ");
                    x.append(extensionInfo.descriptor.getLiteType());
                    throw new IllegalStateException(x.toString());
                case 12:
                    arrayList = new ArrayList<>();
                    reader.readUInt32List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 13:
                    ArrayList arrayList3 = new ArrayList();
                    reader.readEnumList(arrayList3);
                    arrayList2 = new ArrayList();
                    Iterator it = arrayList3.iterator();
                    while (it.hasNext()) {
                        int intValue = ((Integer) it.next()).intValue();
                        Descriptors.EnumValueDescriptor findValueByNumber = extensionInfo.descriptor.getEnumType().findValueByNumber(intValue);
                        if (findValueByNumber != null) {
                            arrayList2.add(findValueByNumber);
                        } else {
                            ub = (UB) SchemaUtil.storeUnknownEnum(number, intValue, ub, unknownFieldSchema);
                        }
                    }
                    break;
                case 14:
                    arrayList = new ArrayList<>();
                    reader.readSFixed32List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 15:
                    arrayList = new ArrayList<>();
                    reader.readSFixed64List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 16:
                    arrayList = new ArrayList<>();
                    reader.readSInt32List(arrayList);
                    arrayList2 = arrayList;
                    break;
                case 17:
                    arrayList = new ArrayList<>();
                    reader.readSInt64List(arrayList);
                    arrayList2 = arrayList;
                    break;
            }
            fieldSet.setField(extensionInfo.descriptor, arrayList2);
        } else {
            Object obj2 = null;
            if (extensionInfo.descriptor.getLiteType() == WireFormat.FieldType.ENUM) {
                int readInt32 = reader.readInt32();
                obj2 = extensionInfo.descriptor.getEnumType().findValueByNumber(readInt32);
                if (obj2 == null) {
                    return (UB) SchemaUtil.storeUnknownEnum(number, readInt32, ub, unknownFieldSchema);
                }
            } else {
                switch (extensionInfo.descriptor.getLiteType().ordinal()) {
                    case 0:
                        obj2 = Double.valueOf(reader.readDouble());
                        break;
                    case 1:
                        obj2 = Float.valueOf(reader.readFloat());
                        break;
                    case 2:
                        obj2 = Long.valueOf(reader.readInt64());
                        break;
                    case 3:
                        obj2 = Long.valueOf(reader.readUInt64());
                        break;
                    case 4:
                        obj2 = Integer.valueOf(reader.readInt32());
                        break;
                    case 5:
                        obj2 = Long.valueOf(reader.readFixed64());
                        break;
                    case 6:
                        obj2 = Integer.valueOf(reader.readFixed32());
                        break;
                    case 7:
                        obj2 = Boolean.valueOf(reader.readBool());
                        break;
                    case 8:
                        obj2 = reader.readString();
                        break;
                    case 9:
                        obj2 = reader.readGroup(extensionInfo.defaultInstance.getClass(), extensionRegistryLite);
                        break;
                    case 10:
                        obj2 = reader.readMessage(extensionInfo.defaultInstance.getClass(), extensionRegistryLite);
                        break;
                    case 11:
                        obj2 = reader.readBytes();
                        break;
                    case 12:
                        obj2 = Integer.valueOf(reader.readUInt32());
                        break;
                    case 13:
                        throw new IllegalStateException("Shouldn't reach here.");
                    case 14:
                        obj2 = Integer.valueOf(reader.readSFixed32());
                        break;
                    case 15:
                        obj2 = Long.valueOf(reader.readSFixed64());
                        break;
                    case 16:
                        obj2 = Integer.valueOf(reader.readSInt32());
                        break;
                    case 17:
                        obj2 = Long.valueOf(reader.readSInt64());
                        break;
                }
            }
            if (extensionInfo.descriptor.isRepeated()) {
                fieldSet.addRepeatedField(extensionInfo.descriptor, obj2);
            } else {
                int ordinal = extensionInfo.descriptor.getLiteType().ordinal();
                if ((ordinal == 9 || ordinal == 10) && (field = fieldSet.getField(extensionInfo.descriptor)) != null) {
                    obj2 = Internal.mergeMessage(field, obj2);
                }
                fieldSet.setField(extensionInfo.descriptor, obj2);
            }
        }
        return ub;
    }

    @Override // com.google.protobuf.ExtensionSchema
    public void parseLengthPrefixedMessageSetItem(Reader reader, Object obj, ExtensionRegistryLite extensionRegistryLite, FieldSet<Descriptors.FieldDescriptor> fieldSet) {
        ExtensionRegistry.ExtensionInfo extensionInfo = (ExtensionRegistry.ExtensionInfo) obj;
        if (ExtensionRegistryLite.isEagerlyParseMessageSets()) {
            fieldSet.setField(extensionInfo.descriptor, reader.readMessage(extensionInfo.defaultInstance.getClass(), extensionRegistryLite));
            return;
        }
        fieldSet.setField(extensionInfo.descriptor, new LazyField(extensionInfo.defaultInstance, extensionRegistryLite, reader.readBytes()));
    }

    @Override // com.google.protobuf.ExtensionSchema
    public void parseMessageSetItem(ByteString byteString, Object obj, ExtensionRegistryLite extensionRegistryLite, FieldSet<Descriptors.FieldDescriptor> fieldSet) {
        ExtensionRegistry.ExtensionInfo extensionInfo = (ExtensionRegistry.ExtensionInfo) obj;
        Message buildPartial = extensionInfo.defaultInstance.newBuilderForType().buildPartial();
        if (ExtensionRegistryLite.isEagerlyParseMessageSets()) {
            BinaryReader newInstance = BinaryReader.newInstance(ByteBuffer.wrap(byteString.toByteArray()), true);
            Protobuf.getInstance().mergeFrom(buildPartial, newInstance, extensionRegistryLite);
            fieldSet.setField(extensionInfo.descriptor, buildPartial);
            if (newInstance.getFieldNumber() != Integer.MAX_VALUE) {
                throw InvalidProtocolBufferException.invalidEndTag();
            }
            return;
        }
        fieldSet.setField(extensionInfo.descriptor, new LazyField(extensionInfo.defaultInstance, extensionRegistryLite, byteString));
    }

    @Override // com.google.protobuf.ExtensionSchema
    public void serializeExtension(Writer writer, Map.Entry<?, ?> entry) {
        Descriptors.FieldDescriptor fieldDescriptor = (Descriptors.FieldDescriptor) entry.getKey();
        if (fieldDescriptor.isRepeated()) {
            switch (fieldDescriptor.getLiteType().ordinal()) {
                case 0:
                    SchemaUtil.writeDoubleList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 1:
                    SchemaUtil.writeFloatList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 2:
                    SchemaUtil.writeInt64List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 3:
                    SchemaUtil.writeUInt64List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 4:
                    SchemaUtil.writeInt32List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 5:
                    SchemaUtil.writeFixed64List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 6:
                    SchemaUtil.writeFixed32List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 7:
                    SchemaUtil.writeBoolList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 8:
                    SchemaUtil.writeStringList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer);
                    return;
                case 9:
                    SchemaUtil.writeGroupList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer);
                    return;
                case 10:
                    SchemaUtil.writeMessageList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer);
                    return;
                case 11:
                    SchemaUtil.writeBytesList(fieldDescriptor.getNumber(), (List) entry.getValue(), writer);
                    return;
                case 12:
                    SchemaUtil.writeUInt32List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 13:
                    ArrayList arrayList = new ArrayList();
                    for (Descriptors.EnumValueDescriptor enumValueDescriptor : (List) entry.getValue()) {
                        arrayList.add(Integer.valueOf(enumValueDescriptor.getNumber()));
                    }
                    SchemaUtil.writeInt32List(fieldDescriptor.getNumber(), arrayList, writer, fieldDescriptor.isPacked());
                    return;
                case 14:
                    SchemaUtil.writeSFixed32List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 15:
                    SchemaUtil.writeSFixed64List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 16:
                    SchemaUtil.writeSInt32List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                case 17:
                    SchemaUtil.writeSInt64List(fieldDescriptor.getNumber(), (List) entry.getValue(), writer, fieldDescriptor.isPacked());
                    return;
                default:
                    return;
            }
        }
        switch (fieldDescriptor.getLiteType().ordinal()) {
            case 0:
                writer.writeDouble(fieldDescriptor.getNumber(), ((Double) entry.getValue()).doubleValue());
                return;
            case 1:
                writer.writeFloat(fieldDescriptor.getNumber(), ((Float) entry.getValue()).floatValue());
                return;
            case 2:
                writer.writeInt64(fieldDescriptor.getNumber(), ((Long) entry.getValue()).longValue());
                return;
            case 3:
                writer.writeUInt64(fieldDescriptor.getNumber(), ((Long) entry.getValue()).longValue());
                return;
            case 4:
                writer.writeInt32(fieldDescriptor.getNumber(), ((Integer) entry.getValue()).intValue());
                return;
            case 5:
                writer.writeFixed64(fieldDescriptor.getNumber(), ((Long) entry.getValue()).longValue());
                return;
            case 6:
                writer.writeFixed32(fieldDescriptor.getNumber(), ((Integer) entry.getValue()).intValue());
                return;
            case 7:
                writer.writeBool(fieldDescriptor.getNumber(), ((Boolean) entry.getValue()).booleanValue());
                return;
            case 8:
                writer.writeString(fieldDescriptor.getNumber(), (String) entry.getValue());
                return;
            case 9:
                writer.writeGroup(fieldDescriptor.getNumber(), entry.getValue());
                return;
            case 10:
                writer.writeMessage(fieldDescriptor.getNumber(), entry.getValue());
                return;
            case 11:
                writer.writeBytes(fieldDescriptor.getNumber(), (ByteString) entry.getValue());
                return;
            case 12:
                writer.writeUInt32(fieldDescriptor.getNumber(), ((Integer) entry.getValue()).intValue());
                return;
            case 13:
                writer.writeInt32(fieldDescriptor.getNumber(), ((Descriptors.EnumValueDescriptor) entry.getValue()).getNumber());
                return;
            case 14:
                writer.writeSFixed32(fieldDescriptor.getNumber(), ((Integer) entry.getValue()).intValue());
                return;
            case 15:
                writer.writeSFixed64(fieldDescriptor.getNumber(), ((Long) entry.getValue()).longValue());
                return;
            case 16:
                writer.writeSInt32(fieldDescriptor.getNumber(), ((Integer) entry.getValue()).intValue());
                return;
            case 17:
                writer.writeSInt64(fieldDescriptor.getNumber(), ((Long) entry.getValue()).longValue());
                return;
            default:
                return;
        }
    }

    @Override // com.google.protobuf.ExtensionSchema
    public void setExtensions(Object obj, FieldSet<Descriptors.FieldDescriptor> fieldSet) {
        UnsafeUtil.putObject(obj, EXTENSION_FIELD_OFFSET, fieldSet);
    }
}